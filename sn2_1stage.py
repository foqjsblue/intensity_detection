import os
import glob
import argparse
import numpy as np
import torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pickle

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
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
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml', help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data', help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def is_point_inside_box(point, box):
    box_center = box[:3]
    box_dims = box[3:6]
    box_yaw = box[6]
    l, w, h = box_dims

    translated_point = point - box_center

    R = np.array([[np.cos(box_yaw), -np.sin(box_yaw), 0],
                  [np.sin(box_yaw), np.cos(box_yaw), 0],
                  [0, 0, 1]])

    rotated_point = np.dot(R, translated_point)

    is_inside = (abs(rotated_point[0]) <= l / 2) and (abs(rotated_point[1]) <= w / 2) and (abs(rotated_point[2]) <= h / 2)
    return is_inside

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), ext=args.ext, logger=logger)
    total_samples = len(demo_dataset)
    logger.info(f'Total number of samples: \t{total_samples}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    stage1_results = []
    fake_box_count = 0  # Initialize fake box counter
    total_boxes = 0  # Initialize total box counter

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            progress = (idx + 1) / total_samples * 100
            print(f"Processing {idx + 1}/{total_samples} ({progress:.2f}%)")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            class_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            total_boxes += len(bounding_boxes)  # Count the number of boxes

            points_in_boxes = [[] for _ in bounding_boxes]
            for p in data_dict['points']:
                x, y, z, i = p[1].item(), p[2].item(), p[3].item(), p[4].item()
                point = np.array([x, y, z])
                for box_idx, box in enumerate(bounding_boxes):
                    if is_point_inside_box(point, box):
                        points_in_boxes[box_idx].append([x, y, z, i])

            for box_idx, pts in enumerate(points_in_boxes):
                box = bounding_boxes[box_idx]
                box_center = box[:3]
                box_dims = box[3:6]
                box_yaw = box[6]
                num_points = len(pts)

                temppointlist_all = []

                box_points = np.array([[p[0], p[1], p[2], p[3]] for p in pts])

                intensity = np.array([p[3] for p in pts])

                distance = np.linalg.norm(box_center)

                label_idx = class_labels[box_idx] - 1
                class_name = cfg.CLASS_NAMES[label_idx] if label_idx < len(cfg.CLASS_NAMES) else "UNKNOWN"

                for p in pts:
                    temppointlist_all.append(p[3])

                if len(temppointlist_all) > 0:
                    temppointlist_all = [float(i) for i in temppointlist_all]
                    threshold = 0.3922
                    count_greater = sum(i > threshold for i in temppointlist_all)
                    percentage_greater = (count_greater / len(temppointlist_all)) * 100

                    if class_name == 'Car':
                        is_fake = percentage_greater > 60
                    elif class_name == 'Pedestrian':
                        is_fake = percentage_greater > 85
                    elif class_name == 'Cyclist':
                        is_fake = percentage_greater > 65
                    else:
                        is_fake = False
                else:
                    is_fake = False

                if is_fake:
                    fake_box_count += 1
                    
                frame_idx = min(idx, len(demo_dataset.sample_file_list) - 1)
                frame_id = int(os.path.splitext(os.path.basename(demo_dataset.sample_file_list[frame_idx]))[0].split('_')[1])  # 파일명에서 frame_id 추출

                box_info = {
                    'frame_id': frame_id,
                    'file_name': os.path.basename(demo_dataset.sample_file_list[frame_idx]),
                    'box_center': box_center.tolist(),
                    'box_dims': box_dims.tolist(),
                    'box_yaw': box_yaw,
                    'class_label': label_idx,
                    'label': 'fake' if is_fake else 'normal',
                    'num_points_in_box': num_points,  # 추가된 정보
                    'points': pts  # 추가된 정보
                }
                
                # 디버깅용 출력
                #print(f"(Stage 1) Frame ID: {frame_id}, File Name: {demo_dataset.sample_file_list[frame_idx]}, Box Center: {box_center}")

                stage1_results.append(box_info)

    with open(os.path.join(args.data_path, 'stage1_results.pkl'), 'wb') as f:
        pickle.dump(stage1_results, f)

    logger.info(f'First stage classification complete. Total boxes: {total_boxes}')
    print(f'Stage 1 complete. Total boxes: {total_boxes}')

if __name__ == '__main__':
    main()
