import os
import numpy as np
import torch
import argparse
import glob
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import random

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path,
                         logger=logger)
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

        # Remove points where z > -1.6
        points = points[points[:, 2] <= -1.6]

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


def save_points_to_bin(points, output_dir, file_name):
    file_path = os.path.join(output_dir, file_name)
    np.array(points, dtype=np.float32).tofile(file_path)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    total_samples = len(demo_dataset)
    logger.info(f'Total number of samples: \t{total_samples}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    saved_count_folder1 = 0
    saved_count_folder2 = 0
    saved_count_folder3 = 0
    saved_count_folder4 = 0
    max_files_per_folder = 500

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            if (saved_count_folder1 >= max_files_per_folder and
                    saved_count_folder2 >= max_files_per_folder and
                    saved_count_folder3 >= max_files_per_folder and
                    saved_count_folder4 >= max_files_per_folder):
                break

            progress = (idx + 1) / total_samples * 100
            print(f"Processing {idx + 1}/{total_samples} ({progress:.2f}%)")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            class_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            file_name = demo_dataset.sample_file_list[idx]
            file_name = os.path.basename(file_name)

            points_in_boxes = [[] for _ in bounding_boxes]
            for p in data_dict['points']:
                x, y, z, i = p[1].item(), p[2].item(), p[3].item(), p[4].item()
                for idx, box in enumerate(bounding_boxes):
                    if is_point_inside_box(np.array([x, y, z]), box):
                        points_in_boxes[idx].append([x, y, z, i])

            # Filter bounding boxes with at least 10 points
            valid_boxes = [pts for pts in points_in_boxes if len(pts) >= 10]

            # Randomly select one bounding box if there are any valid boxes
            if valid_boxes:
                selected_box = random.choice(valid_boxes)
                if saved_count_folder1 < max_files_per_folder:
                    output_dir = 'normal_one_sc'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    save_points_to_bin(selected_box, output_dir, file_name)
                    saved_count_folder1 += 1
                elif saved_count_folder2 < max_files_per_folder:
                    output_dir = 'normal_two_sc'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    save_points_to_bin(selected_box, output_dir, file_name)
                    saved_count_folder2 += 1
                elif saved_count_folder3 < max_files_per_folder:
                    output_dir = 'normal_three_sc'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    save_points_to_bin(selected_box, output_dir, file_name)
                    saved_count_folder3 += 1
                elif saved_count_folder4 < max_files_per_folder:
                    output_dir = 'normal_four_sc'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    save_points_to_bin(selected_box, output_dir, file_name)
                    saved_count_folder4 += 1

    logger.info('Processing Complete.')


if __name__ == '__main__':
    main()
