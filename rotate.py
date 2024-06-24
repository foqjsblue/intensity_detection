import numpy as np
import os
import math

def rotate_point_cloud(data):
    """ Apply a random yaw rotation around the z-axis center to the point cloud """
    points = data[:, :3]  # Select only the x, y, z coordinates
    rotation_angle = np.random.rand() * 2 * math.pi  # A random angle between 0 and 2π
    cos_val = np.cos(rotation_angle)
    sin_val = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                [sin_val, cos_val, 0],
                                [0, 0, 1]])
    rotated_points = np.dot(points, rotation_matrix)
    data[:, :3] = rotated_points  # Replace with rotated points
    return data

def process_and_save_files(input_folder, output_folder):
    """ Rotate all point cloud .bin files in the given folder and save them to a new folder """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if file_path.endswith('.bin'):
            data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            rotated_data = rotate_point_cloud(data)
            output_file_path = os.path.join(output_folder, filename)
            # Save the rotated data as .bin files
            rotated_data.tofile(output_file_path)

input_folder = '/mnt/d/kitti_box/box_cyc/training/s1/cyc_train_rd02710_s1'
output_folder = '/mnt/d/kitti_box/box_cyc/training/s1/cyc_train_rd02710_s1_rotate2'

process_and_save_files(input_folder, output_folder)
