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
import shutil

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
    logger.info("Sample file list:")
    for file in demo_dataset.sample_file_list:
        logger.info(file)

    #logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    total_samples = len(demo_dataset)
    logger.info(f'Total number of samples: \t{total_samples}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()




#####################################################################################################################

    temppointlist = []
    total_bounding_boxes = 0

    save_dir = 'fake_car_pgd_pp'
    class_save_dir = Path(save_dir)
    class_save_dir.mkdir(parents=True, exist_ok=True)

    car_counter = 0

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            class_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            points_in_boxes = [[] for _ in bounding_boxes]
            for p in data_dict['points']:
                x, y, z, i = p[1].item(), p[2].item(), p[3].item(), p[4].item()
                for box_idx, box in enumerate(bounding_boxes):
                    if is_point_inside_box(np.array([x, y, z]), box):
                        points_in_boxes[box_idx].append([x, y, z, i])

            for box_idx, pts in enumerate(points_in_boxes):
                class_label_idx = class_labels[box_idx] - 1
                class_name = cfg.CLASS_NAMES[class_label_idx] if class_label_idx < len(cfg.CLASS_NAMES) else "UNKNOWN"

                if class_name == "Car" and len(pts) >= 10: ##
                    file_name_with_counter = f"merged_data_car_{car_counter}.bin"
                    dest_file_path = class_save_dir / file_name_with_counter

                    points = np.array(pts, dtype=np.float32)
                    points.tofile(dest_file_path)

                    logger.info(f"Saved {file_name_with_counter} to {dest_file_path}")
                    car_counter += 1

    print(f"Total number of bounding boxes: {total_bounding_boxes}")
    logger.info('Attack Detection Complete.')
    return box_data


if __name__ == '__main__':
    main()
