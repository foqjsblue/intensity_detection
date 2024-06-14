import os
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.stats import skew, kurtosis
import argparse
import glob
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.threshold = 0.3

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4) if self.ext == '.bin' else np.load(self.sample_file_list[index])
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml', help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data', help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='output', help='specify the output directory')
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

def is_point_inside_box_vectorized(points, box):
    center = box[:3]
    hwl = box[3:6] / 2  # half dimensions: half-length, half-width, half-height

    # Create rotation matrix for Z-axis rotation
    theta = box[6]  # rotation angle in radians
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Transform points to the box's coordinate frame
    points_local = np.dot(points[:, :3] - center, R)

    # Check if the points are inside the bounding box
    inside_box = np.all(np.abs(points_local) <= hwl, axis=1)
    return inside_box

def cluster_points_with_dbscan(points, eps=0.3, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, num_clusters

class ShadowCatcher:
    def __init__(self, threshold):
        self.threshold = threshold

    def shadow_region_proposal(self, points, box, lidar_position=np.array([0, 0, 1.8])):
        center = box[:3]
        dimensions = box[3:6]
        half_dimensions = dimensions / 2
        theta = box[6]

        # Z-axis rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Calculate shadow length
        object_height = dimensions[2]
        dobj = np.linalg.norm(center[:2] - lidar_position[:2])
        shadow_length = dobj * object_height / (lidar_position[2] - object_height)

        # Calculate shadow region bounds
        shadow_points = []
        for point in points:
            translated_point = point - center
            local_point = np.dot(rotation_matrix.T, translated_point)
            if abs(local_point[0]) <= half_dimensions[0] and abs(local_point[1]) <= half_dimensions[1]:
                distance = np.linalg.norm(point[:2] - lidar_position[:2])
                if distance <= shadow_length:
                    shadow_points.append(point)

        return np.array(shadow_points)

    def genuine_shadow_verification(self, shadow_region, box):
        if len(shadow_region) == 0:
            return 1.0  # Return a high score indicating no genuine shadow

        distances = self.calculate_distances(shadow_region, box)
        score = self.calculate_anomaly_score(distances)
        return score

    def calculate_distances(self, shadow_region, box):
        center = box[:3]
        half_dimensions = box[3:6] / 2
        theta = box[6]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        translated_points = shadow_region - center
        rotated_points = translated_points.dot(rotation_matrix.T)

        distances = {
            'start_line': np.abs(rotated_points[:, 0] + half_dimensions[0]),
            'end_line': np.abs(rotated_points[:, 0] - half_dimensions[0]),
            'center_line': np.abs(rotated_points[:, 1]),
            'boundary_line': np.abs(rotated_points[:, 1]) - half_dimensions[1]
        }

        return distances

    def calculate_anomaly_score(self, distances):
        scores = {
            'start_line': skew(distances['start_line']) + kurtosis(distances['start_line']),
            'end_line': skew(distances['end_line']) + kurtosis(distances['end_line']),
            'center_line': skew(distances['center_line']) + kurtosis(distances['center_line']),
            'boundary_line': skew(distances['boundary_line']) + kurtosis(distances['boundary_line'])
        }
        total_score = np.mean(list(scores.values()))
        return total_score

    def adversarial_shadow_classification(self, shadow_region):
        if len(shadow_region) == 0:
            return False

        clustering = DBSCAN(eps=0.3, min_samples=10).fit(shadow_region)
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return num_clusters > 1

    def verify_shadow(self, points, box):
        shadow_region = self.shadow_region_proposal(points, box)
        score = self.genuine_shadow_verification(shadow_region, box)

        if np.isnan(score):
            return False, score

        is_genuine = score <= self.threshold

        if not is_genuine:
            adversarial_class = self.adversarial_shadow_classification(shadow_region)
            return adversarial_class, score  # Return True if it's a genuine shadow, False otherwise

        return is_genuine, score

def save_points_to_bin(points, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    points.tofile(file_path)

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), ext=args.ext, logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    total_samples = len(demo_dataset)
    logger.info(f'Total number of samples: \t{total_samples}')

    shadow_catcher = ShadowCatcher(threshold=0.7)
    total_fake_shadows = 0
    total_processed_shadows = 0

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):

            processed_shadows_per_file = 0
            fake_shadows_per_file = 0

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            points = data_dict['points'].cpu().numpy()
            points_to_keep = points.copy()

            for box in bounding_boxes:
                if len(box) != 7:
                    pass
                else:
                    inside_box = is_point_inside_box_vectorized(points[:, 1:4], box)
                    shadow_points = points[inside_box]

                    if len(shadow_points) >= 10:  # Only process if there are enough points
                        is_genuine, score = shadow_catcher.verify_shadow(shadow_points[:, 1:4], box)
                        if not is_genuine:  # Explicit comparison with False
                            points_to_keep = points_to_keep[~inside_box]
                            fake_shadows_per_file += 1

                        processed_shadows_per_file += 1

            total_fake_shadows += fake_shadows_per_file  # Increment total fake shadows counter
            fake_percentage = (
                                          fake_shadows_per_file / processed_shadows_per_file) * 100 if processed_shadows_per_file > 0 else 0
            logger.info(
                f'File {idx + 1}/{len(demo_dataset)}: Processed {processed_shadows_per_file} shadows, Fake shadows: {fake_shadows_per_file} ({fake_percentage:.2f}%)')
            total_processed_shadows += processed_shadows_per_file

            output_file_name = os.path.basename(demo_dataset.sample_file_list[idx])
            save_points_to_bin(points_to_keep[:, 1:], output_dir,
                               output_file_name)  # Save only the x, y, z, intensity columns

    if total_processed_shadows > 0:
        fake_percentage = (total_fake_shadows / total_processed_shadows) * 100
    else:
        fake_percentage = 0

    logger.info(
        f'Attack Detection Complete. Total processed shadows: {total_processed_shadows}, '
        f'Total fake shadows: {total_fake_shadows} '
        f'({fake_percentage:.2f}%)'
    )

if __name__ == '__main__':
    main()

