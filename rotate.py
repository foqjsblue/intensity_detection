import numpy as np
import os
import math

def rotate_point_cloud(data):
    """ 포인트 클라우드에 z축 중심 랜덤 yaw 회전을 적용합니다. """
    points = data[:, :3]  # x, y, z 좌표만 선택
    rotation_angle = np.random.rand() * 2 * math.pi  # 0에서 2파이 사이의 랜덤 각도
    cos_val = np.cos(rotation_angle)
    sin_val = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                [sin_val, cos_val, 0],
                                [0, 0, 1]])
    rotated_points = np.dot(points, rotation_matrix)  # 회전 적용
    data[:, :3] = rotated_points  # 회전된 포인트로 대체
    return data

def process_and_save_files(input_folder, output_folder):
    """ 주어진 폴더의 모든 포인트 클라우드 .bin 파일을 회전시키고 새 폴더에 저장합니다. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if file_path.endswith('.bin'):  # .bin 파일만 처리
            # 포인트 클라우드 데이터 읽기
            data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            rotated_data = rotate_point_cloud(data)
            output_file_path = os.path.join(output_folder, filename)
            # 회전된 데이터를 .bin 파일로 저장
            rotated_data.tofile(output_file_path)

# 폴더 경로 설정
input_folder = '/mnt/d/kitti_box/box_cyc/training/s1/cyc_train_rd02710_s1'
output_folder = '/mnt/d/kitti_box/box_cyc/training/s1/cyc_train_rd02710_s1_rotate2'

# 파일 처리 실행
process_and_save_files(input_folder, output_folder)

