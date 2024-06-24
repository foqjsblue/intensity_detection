import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
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

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    mlab.options.offscreen = True
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False


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


def find_nearest_neighbors(points, k=10):
    if len(points) < k:
        return []

    points_array = np.array(points)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points_array[:, :3])
    distances, indices = nbrs.kneighbors(points_array[:, :3])
    return indices

def count_similar_intensity_points(points, neighbors_indices, intensity_threshold=0.01):
    count_knn = 0
    total_intensity = 0

    for i_knn, point in enumerate(points):
        if i_knn < len(neighbors_indices):
            neighbor_idxs = neighbors_indices[i_knn]
            neighbors = [points[idx] for idx in neighbor_idxs]

            if all(abs(point[3] - neighbor[3]) <= intensity_threshold for neighbor in neighbors):
                count_knn += 1
                total_intensity += point[3]

    avg_intensity_knn = total_intensity / count_knn if count_knn > 0 else 0
    return count_knn, avg_intensity_knn

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
    total_samples = len(demo_dataset)
    logger.info(f'Total number of samples: \t{total_samples}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()




#####################################################################################################################

    total_bounding_boxes = 0
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
                if i != 0:
                    for idx, box in enumerate(bounding_boxes):
                        if is_point_inside_box(np.array([x, y, z]), box):
                            points_in_boxes[idx].append([x, y, z, i])

            for idx, points_in_box in enumerate(points_in_boxes):
                label_idx = class_labels[idx] - 1
                class_name = cfg.CLASS_NAMES[label_idx]
                box_center = bounding_boxes[idx][:3]
                distance = math.sqrt(box_center[0] ** 2 + box_center[1] ** 2 + box_center[2] ** 2)

                if class_name in ['Car', 'Pedestrian', 'Cyclist'] and len(points_in_box) >= 10 and distance <= 20 :
                    f_name = f'{class_name.lower()}_{global_box_counters[class_name]}.bin'
                    output_dir = f'../box_{class_name.lower()}'

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    save_points_to_bin(points_in_box, output_dir, f_name)
                    global_box_counters[class_name] += 1

        print(distances.keys())

    print(f"Total number of bounding boxes: {total_bounding_boxes}")
    #print(f"▶ Number of bounding boxes saved: {global_box_counter - 1}")
    for class_name, counter in global_box_counters.items():
        print(f"▶ Number of {class_name} bounding boxes saved: {counter - 1}")
    logger.info('Attack Detection Complete.')
    return box_data


if __name__ == '__main__':
    main()
