import os
import numpy as np
import torch
import argparse
from pathlib import Path
import glob
from collections import defaultdict
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
        self.sample_file_list = sorted(glob.glob(str(self.root_path / f'*{self.ext}')))

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
    parser.add_argument('--data_path', type=str, required=True,
                        help='specify the point cloud data directory before attack')
    parser.add_argument('--data_path_2', type=str, required=True,
                        help='specify the point cloud data directory after attack')
    parser.add_argument('--data_path_3', type=str, required=True,
                        help='specify the point cloud data directory after detection')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--show_demo', action='store_true', default=True, help='Display demo visualization')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def count_objects(data_dict, model):
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict)
    return pred_dicts[0]['pred_boxes'].cpu().numpy()


def calculate_tp_fp(objects_normal, objects_attacked, objects_detected, distance_threshold=0.0):
    # 공격 후 중심점 개수 - 탐지 후 중심점 개수 = 탐지법이 사라지게 한 바운딩 박스 개수 (FP의 최댓값)
    max_fp = len(objects_attacked) - len(objects_detected)

    # 공격 후 중심점과 공격 전 중심점 비교: 공격 후에 생긴 중심점
    added_centroids = [box[:3] for box in objects_attacked if
                       not any(np.allclose(box[:3], obj[:3], atol=distance_threshold) for obj in objects_normal)]

    # TP: 공격 후에 생긴 중심점이 탐지 후에 사라졌다면 TP 증가
    tp = sum(1 for centroid in added_centroids if
             not any(np.allclose(centroid, det[:3], atol=distance_threshold) for det in objects_detected))

    tp = min(tp, 1)  # TP가 1을 초과하지 않도록 설정

    # 최종 FP = 탐지법이 사라지게 한 바운딩 박스 개수 - TP
    fp = max_fp - tp
    fp = max(fp, 0)  # FP가 음수가 되지 않도록 설정

    return tp, fp, len(objects_detected), max_fp


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    def create_dataset(data_path, ext, label):
        root_path = Path(data_path)
        sample_file_list = sorted(glob.glob(str(root_path / f'*{ext}')))
        return DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=root_path, ext=ext, logger=logger
        )

    dataset_normal = create_dataset(args.data_path, args.ext, 'normal')
    dataset_attacked = create_dataset(args.data_path_2, args.ext, 'attacked')
    dataset_detected = create_dataset(args.data_path_3, args.ext, 'detected')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_normal)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    file_groups = defaultdict(lambda: {'normal': None, 'attacked': None, 'detected': None})

    def add_files_to_group(sample_file_list, label):
        for file_path in sample_file_list:
            file_name = os.path.basename(file_path)
            base_name = file_name.split('_')[1].split('.')[0]
            file_groups[base_name][label] = file_path

    add_files_to_group(dataset_normal.sample_file_list, 'normal')
    add_files_to_group(dataset_attacked.sample_file_list, 'attacked')
    add_files_to_group(dataset_detected.sample_file_list, 'detected')

    # 디버깅 출력: 파일 그룹 확인
    for base_name, paths in file_groups.items():
        logger.info(f"File group: {base_name}, Paths: {paths}")

    total_TP = 0
    total_FP = 0
    total_objects = 0
    total_detected_objects = 0

    with torch.no_grad():
        for base_name, paths in file_groups.items():
            if paths['normal'] and paths['attacked'] and paths['detected']:
                normal_idx = dataset_normal.sample_file_list.index(paths['normal'])
                attacked_idx = dataset_attacked.sample_file_list.index(paths['attacked'])
                detected_idx = dataset_detected.sample_file_list.index(paths['detected'])

                data_dict_normal = dataset_normal[normal_idx]
                data_dict_attacked = dataset_attacked[attacked_idx]
                data_dict_detected = dataset_detected[detected_idx]

                data_dict_normal = dataset_normal.collate_batch([data_dict_normal])
                data_dict_attacked = dataset_attacked.collate_batch([data_dict_attacked])
                data_dict_detected = dataset_detected.collate_batch([data_dict_detected])

                load_data_to_gpu(data_dict_normal)
                load_data_to_gpu(data_dict_attacked)
                load_data_to_gpu(data_dict_detected)

                objects_normal = count_objects(data_dict_normal, model)
                objects_attacked = count_objects(data_dict_attacked, model)
                objects_detected = count_objects(data_dict_detected, model)

                tp, fp, num_detected, max_fp = calculate_tp_fp(objects_normal, objects_attacked, objects_detected)

                logger.info(
                    f"File {base_name}: TP = {tp}, FP = {fp}, Detected Objects = {num_detected}, Max FP = {max_fp}")

                total_TP += tp
                total_FP += fp
                total_objects += len(objects_normal)
                total_detected_objects += num_detected

                logger.info(
                    f"Current totals - TP: {total_TP}, FP: {total_FP}, Total objects: {total_objects}, Total detected objects: {total_detected_objects}")

    TP_percentage = (total_TP / 50) * 100 if 50 > 0 else 0
    FP_percentage = (total_FP / total_objects) * 100 if total_objects > 0 else 0

    logger.info(
        f"Overall: TP = {total_TP} ({TP_percentage:.2f}%), FP = {total_FP} ({FP_percentage:.2f}%), Detected Objects = {total_detected_objects}")
    print(
        f"Overall: TP = {total_TP} ({TP_percentage:.2f}%), FP = {total_FP} ({FP_percentage:.2f}%), Detected Objects = {total_detected_objects}")


if __name__ == '__main__':
    main()
