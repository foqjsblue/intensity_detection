import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
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
import random
import logging

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
    parser.add_argument('--box_path', type=str, default=None,
                        help='Specify the path of the folder containing additional point cloud data files to be merged')
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_points_to_bin(points, output_dir, file_name):
    try:
        file_path = os.path.join(output_dir, file_name)
        np.array(points, dtype=np.float32).tofile(file_path)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")


def load_point_cloud_from_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def merge_point_clouds(background_pc, additional_pc):
    return np.concatenate([background_pc, additional_pc], axis=0)


def adjust_intensity_values(point_cloud, min_intensity=0.18, max_intensity=1.0):
    adjusted_pc = np.copy(point_cloud)
    random_intensities = np.random.uniform(min_intensity, max_intensity, size=(adjusted_pc.shape[0],))
    adjusted_pc[:, 3] = random_intensities
    return adjusted_pc


def filter_points(points):
    def calculate_angle_and_intensity(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arcsin(z / r)
        return theta, phi

    unique_points = {}
    for point in points:
        x, y, z, intensity = point
        theta, phi = calculate_angle_and_intensity(x, y, z)

        angle_key = (round(theta, 5), round(phi, 5))
        if angle_key not in unique_points or unique_points[angle_key][3] < intensity:
            unique_points[angle_key] = point

    return np.array(list(unique_points.values()))


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    box_data = []

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    base_point_cloud_path = args.data_path
    box_path = args.box_path

    if args.box_path:
        base_files = glob.glob(os.path.join(args.data_path, '*.bin'))
        box_files = glob.glob(os.path.join(args.box_path, '*.bin'))

        min_len = min(len(base_files), len(box_files))

        for idx in range(min_len):
            base_point_cloud = load_point_cloud_from_bin(base_files[idx])
            box_point_cloud = load_point_cloud_from_bin(box_files[idx])

            box_point_cloud = adjust_intensity_values(box_point_cloud)

            merged_point_cloud = merge_point_clouds(base_point_cloud, box_point_cloud)

            filtered_points = filter_points(merged_point_cloud)

            output_directory = 'fgsm_sn2_pp_merged'
            os.makedirs(output_directory, exist_ok=True)

            output_filename = f'merged_{idx}.bin'
            output_filepath = os.path.join(output_directory, output_filename)
            save_points_to_bin(filtered_points, output_directory, output_filename)

            logger.info(f'Merged file saved to {output_filepath}')

            # Save the original base file separately in 'bg_rd_sc' folder
            bg_output_directory = 'fgsm_sn2_pp_bg'
            os.makedirs(bg_output_directory, exist_ok=True)
            bg_output_filepath = os.path.join(bg_output_directory, output_filename)
            save_points_to_bin(base_point_cloud, bg_output_directory, output_filename)

            logger.info(f'Original base file saved to {bg_output_filepath}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    temppointlist = []
    total_bounding_boxes = 0

    with torch.no_grad():
        distances = {}
        intensities = {}

        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Sample index: \t{idx + 1}')
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


        print(distances.keys())

    print(f"Total number of bounding boxes: {total_bounding_boxes}")
    logger.info('Attack Detection Complete.')
    return box_data


if __name__ == '__main__':
    main()
