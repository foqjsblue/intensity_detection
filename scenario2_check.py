import os
import shutil
from pathlib import Path
import numpy as np
import torch
import argparse
import glob
import math
from scipy import stats

from sklearn.neighbors import NearestNeighbors
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path,
                         logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = list(self.root_path.glob(f'*{self.ext}'))

        if not self.sample_file_list:
            raise ValueError(f"No data files found in {self.root_path} with extension {self.ext}")

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
    parser.add_argument('--data_path_2', type=str, help='specify the second point cloud data file or directory')
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


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    dataset_1 = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    dataset_2 = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path_2), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_1)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    saved_count = 0
    output_dir = 'input_sn2_pp_fgsm'
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset_1)):
            file_name_1 = os.path.basename(dataset_1.sample_file_list[idx])
            file_name_2 = os.path.basename(dataset_2.sample_file_list[idx])

            if file_name_1 != file_name_2:
                logger.warning(f"File names do not match: {file_name_1} and {file_name_2}")
                continue

            data_dict_1 = dataset_1[idx]
            data_dict_2 = dataset_2[idx]

            # Ensure data_dict has all necessary keys and correct formats
            data_dict_1['batch_size'] = 1
            data_dict_2['batch_size'] = 1

            data_dict_1 = dataset_1.collate_batch([data_dict_1])
            data_dict_2 = dataset_2.collate_batch([data_dict_2])

            load_data_to_gpu(data_dict_1)
            load_data_to_gpu(data_dict_2)

            pred_dicts_1, _ = model.forward(data_dict_1)
            pred_dicts_2, _ = model.forward(data_dict_2)

            num_objects_1 = len(pred_dicts_1[0]['pred_boxes'])
            num_objects_2 = len(pred_dicts_2[0]['pred_boxes'])

            if num_objects_2 == num_objects_1 + 1:
                output_path = os.path.join(output_dir, file_name_2)
                shutil.copy2(dataset_2.sample_file_list[idx], output_path)
                saved_count += 1

                if saved_count >= 50:
                    break

    logger.info(f'Saved {saved_count} files with one additional object.')


if __name__ == '__main__':
    main()
