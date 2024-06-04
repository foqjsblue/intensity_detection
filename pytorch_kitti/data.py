import os
import numpy as np
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    def __init__(self, directories, num_points, labels, partition='training'):

        self.num_points = num_points
        self.files = []
        self.labels = []
        self.partition = partition
        #self.label_mapping = {'Pedestrian': 0, 'Fake Pedestrian': 1}
        self.label_mapping = {'Normal Object': 0, 'Fake Object': 1}

        for directory, label in zip(directories, labels):
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bin')]
            self.files.extend(files)
            # Map string labels to integers
            self.labels.extend([self.label_mapping[label]] * len(files))

    def load_bin_file(self, file_path):
        print(f"Loading file: {file_path}")
        pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return pointcloud
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        pointcloud = self.pad_pointcloud(pointcloud)
        label = self.labels[idx]

        # Adjust the range of the intensity values (the last column of the point cloud) from 0-1 to 0-255
        pointcloud[:, 3] = pointcloud[:, 3] * 255

        return pointcloud, label

    def __len__(self):
        return len(self.files)

    def pad_pointcloud(self, pointcloud):
        if len(pointcloud) < self.num_points:
            zeros = np.zeros((self.num_points - len(pointcloud), 4), dtype=np.float32)
            pointcloud = np.concatenate([pointcloud, zeros], axis=0)
        return pointcloud


if __name__ == '__main__':
    train_directories = ['kitti_box/box_car/box_car_train', 'kitti_box/box_ped/training/box_ped_train', 'kitti_box/box_cyc/trainingg/box_cyc_train', 'kitti_box/box_car/testing/box_car_test_5t', 'kitti_box/box_ped/testing/box_ped_train_5t', 'kitti_box/box_cyc/training/box_cyc_test_5t']
    train_labels = ['Normal Object', 'Normal Object', 'Normal Object', 'Fake Object', 'Fake Object', 'Fake Object']
    dataset = KITTIDataset(directories=train_directories, labels=train_labels, num_points=256)
    #partition = 'kitti_box/training'

    for data, label in dataset:
        print(data.shape, label)
