import os
os.environ['QT_QPA_PLATFORM'] = 'offsreen'
#os.environ['ETS_TOOLKIT'] = 'wx'
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis

import numpy as np
import torch
import argparse
import glob
import math
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--show_demo', action='store_true', default=True, help='Display demo visualization')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def is_point_inside_box(point, box):
    l, w, h = box[3], box[4], box[5]
    R = np.array([[np.cos(box[6]), -np.sin(box[6]), 0],
                  [np.sin(box[6]), np.cos(box[6]), 0],
                  [0, 0, 1]])
    #R = np.array([[np.cos(box[6]), 0, -np.sin(box[6])],
    #              [np.sin(box[6]), 0, np.cos(box[6])],
    #              [0, 0, 1]])
    rotated_point = np.dot(R.T, point - box[:3])
    return (abs(rotated_point[0]) <= l / 2) and (abs(rotated_point[1]) <= w / 2) and (abs(rotated_point[2]) <= h / 2)

def save_points_to_bin(points, output_dir, file_name):
    file_path = os.path.join(output_dir, file_name)
    np.array(points, dtype=np.float32).tofile(file_path)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    box_data = []
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    #logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    total_samples = len(demo_dataset)  # 전체 샘플의 수
    logger.info(f'Total number of samples: \t{total_samples}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

#####################################################################################################################

    temppointlist = []
    fake_car_count = 0
    real_car_count = 0
    fake_ped_count = 0
    real_ped_count = 0
    fake_cyc_count = 0
    real_cyc_count = 0

    total_bounding_boxes = 0
    fake_car_info = [] 
    fake_ped_info = []
    fake_cyc_info = []

    # 바운딩 박스를 위한 전역 카운터
    global_box_counters = {'Car': 1, 'Pedestrian': 1, 'Cyclist': 1}

    with (((torch.no_grad()))):
        distances = {}
        intensities = {}

        for idx, data_dict in enumerate(demo_dataset):
            #logger.info(f'Sample index: \t{idx + 1}')
            progress = (idx + 1) / total_samples * 100
            print(f"Processing {idx + 1}/{total_samples} ({progress:.2f}%)")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            class_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            total_bounding_boxes += len(bounding_boxes)

            file_name = demo_dataset.sample_file_list[idx]
            file_name = os.path.basename(file_name)

            points_in_boxes = [[] for _ in bounding_boxes]
            for p in data_dict['points']:
                x, y, z, i = p[1].item(), p[2].item(), p[3].item(), p[4].item()
                for idx, box in enumerate(bounding_boxes):
                    if is_point_inside_box(np.array([x, y, z]), box):
                        points_in_boxes[idx].append([x, y, z, i])

            for idx, pts in enumerate(points_in_boxes):

                temppointlist_all = []
                box_points = np.array([[p[0], p[1], p[2], p[3]] for p in pts])
                intensity = np.array([p[3] for p in pts])

                num_points = len(box_points)

                box_center = bounding_boxes[idx][:3]
                distance = math.sqrt(box_center[0] ** 2 + box_center[1] ** 2 + box_center[2] ** 2)

                label_idx = class_labels[idx] - 1
                class_name = cfg.CLASS_NAMES[label_idx] if label_idx < len(cfg.CLASS_NAMES) else "UNKNOWN"

                box_points = []
                for p in pts:
                    x, y, z, i = p
                    if i != 0:
                        box_points.append([x, y, z, i])

                for p in pts :
                    temppointlist_all.append(p[3])

                    
                if not class_name in distances:
                    distances[class_name] = []
                    intensities[class_name] = []
                distances[class_name].append(distance)

                #if len(temppointlist_all) >= 10:
                if len(temppointlist_all) > 0:

                    # Convert string values to float
                    temppointlist_all = [float(i) for i in temppointlist_all]
                    threshold = 0.3922

                    # Count the number of elements greater than and less than or equal to the threshold
                    count_greater = sum(i > threshold for i in temppointlist_all)
                    count_lesser_or_equal = sum(i <= threshold for i in temppointlist_all)

                    #print(f"temppointlist_all:{len(temppointlist_all)}")


                    # Calculate the percentage
                    total_elements = len(temppointlist_all)
                    percentage_greater = (count_greater / total_elements) * 100

                    count_greater, count_lesser_or_equal, percentage_greater

                    if class_name == 'Car' :

                        if percentage_greater <= 60 :
                            fake_car_count += 1
                            f_name = f'fake_car_{global_box_counters["Car"]}.bin'
                            output_dir = f'../box_car_fp_sc'
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            save_points_to_bin(box_points, output_dir, f_name)
                            global_box_counters['Car'] += 1
                        else :
                            real_car_count += 1

                    else :
                        pass

                    if class_name == 'Pedestrian' :

                        if percentage_greater <= 85 :
                            fake_ped_count += 1
                            f_name = f'fake_ped_{global_box_counters["Pedestrian"]}.bin'
                            output_dir = f'../box_ped_fp_sc'
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            save_points_to_bin(box_points, output_dir, f_name)
                            global_box_counters['Pedestrian'] += 1
                        else :
                            real_ped_count += 1

                    else :
                        pass

                    if class_name == 'Cyclist' :

                        if percentage_greater <= 65 :
                            fake_cyc_count += 1
                            f_name = f'fake_cyc_{global_box_counters["Cyclist"]}.bin'
                            output_dir = f'../box_cyc_fp_sc'
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            save_points_to_bin(box_points, output_dir, f_name)
                            global_box_counters['Cyclist'] += 1

                        else :
                            real_cyc_count += 1

                    else :
                        pass

        print(distances.keys())

    car_percentage = (fake_car_count / (fake_car_count + real_car_count)) * 100 if (fake_car_count + real_car_count) > 0 else 0
    ped_percentage = (fake_ped_count / (fake_ped_count + real_ped_count)) * 100 if (fake_ped_count + real_ped_count) > 0 else 0
    cyc_percentage = (fake_cyc_count / (fake_cyc_count + real_cyc_count)) * 100 if (fake_cyc_count + real_cyc_count) > 0 else 0
    total_percentage = ((fake_car_count + fake_ped_count + fake_cyc_count) / (fake_car_count + real_car_count + fake_ped_count + real_ped_count + fake_cyc_count + real_cyc_count)) * 100 if (fake_car_count + real_car_count + fake_ped_count + real_ped_count + fake_cyc_count + real_cyc_count) > 0 else 0

    print(f"Total number of bounding boxes: {total_bounding_boxes}")
    print(f"Number of fake cars: {fake_car_count} / {fake_car_count + real_car_count}({car_percentage:.2f}%)")
    print(f"Number of fake pedestrians: {fake_ped_count} / {fake_ped_count + real_ped_count}({ped_percentage:.2f}%)")
    print(f"Number of fake cyclists: {fake_cyc_count} / {fake_cyc_count + real_cyc_count}({cyc_percentage:.2f}%)")
    print(f"Total number of fake objects: {fake_car_count + fake_ped_count + fake_cyc_count} / {fake_car_count + real_car_count + fake_ped_count + real_ped_count + fake_cyc_count + real_cyc_count}({total_percentage:.2f}%)")
    print(f"Total number of normal objects: {(fake_car_count + real_car_count + fake_ped_count + real_ped_count + fake_cyc_count + real_cyc_count)-(fake_car_count + fake_ped_count + fake_cyc_count)} / {fake_car_count + real_car_count + fake_ped_count + real_ped_count + fake_cyc_count + real_cyc_count}({total_percentage:.2f}%)")
    logger.info('Attack Detection Complete.')
    return box_data




if __name__ == '__main__':
    main()
