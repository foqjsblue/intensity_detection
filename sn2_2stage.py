import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from model import DGCNN

class BoundingBoxDataset(Dataset):
    def __init__(self, data_path, num_points):
        self.data_path = data_path
        self.num_points = num_points
        with open(os.path.join(data_path, 'stage1_results.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        
        # Extract frame_ids from file names
        bin_files = sorted(glob.glob(os.path.join(data_path, 'merged_*.bin')))
        self.file_mapping = {}
        for file in bin_files:
            frame_id = int(os.path.splitext(os.path.basename(file))[0].split('_')[1])
            self.file_mapping[frame_id] = file
        
        self.frame_ids = list(self.file_mapping.keys())
        self.label_mapping = {'Normal Object': 0, 'Fake Object': 1}
        
        # Stage 1의 frame_id 로그 확인 ####
        for item in self.data: ####
            print(f"Stage 1 Frame ID: {item.get('frame_id')}, Box Center: {item['box_center']}") ####
        
        # file_mapping 로그 확인 ####
        #print(f"file_mapping: {self.file_mapping}") ####

    def __len__(self):
        return len(self.data)

    def pad_pointcloud(self, pointcloud):
        if len(pointcloud) < self.num_points:
            zeros = np.zeros((self.num_points - len(pointcloud), 4), dtype=np.float32)
            pointcloud = np.concatenate([pointcloud, zeros], axis=0)
        return pointcloud

    def __getitem__(self, idx):
        item = self.data[idx]
        frame_id = item['frame_id']  # Stage 1의 frame_id를 그대로 사용

        if frame_id not in self.file_mapping:
            raise FileNotFoundError(f"No such file for frame_id {frame_id}. Available frame_ids: {self.frame_ids}")

        if item['label'] != 'normal':  # 'normal' label만 처리 ####
            return None

        points_in_box = np.array(item['points'])
        
        # Ensure points_in_box is at least 2D
        if points_in_box.ndim == 1:
            points_in_box = points_in_box.reshape(-1, 4)

        # Intensity 값 (pointcloud의 마지막 열)의 범위를 0-1에서 0-255로 조정
        points_in_box[:, 3] = points_in_box[:, 3] * 255

        points_in_box = self.pad_pointcloud(points_in_box)

        label = self.label_mapping['Fake Object'] if item['label'] == 'fake' else self.label_mapping['Normal Object']
        return torch.from_numpy(points_in_box).float(), torch.tensor(label).long(), torch.tensor(frame_id).long(), torch.tensor(len(item['points'])).long(), torch.from_numpy(np.concatenate([item['box_center'], item['box_dims'], [item['box_yaw']]])).float(), item['box_center'], item['file_name'], item['points']

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # 'None' 항목 필터링 ####
    if len(batch) == 0:
        return None

    max_size = max([s[0].shape[0] for s in batch])

    padded_data = []
    labels = []
    frame_ids = []
    num_points_in_boxes = []
    boxes = []
    centers = []
    file_names = []
    points_list = []

    for item in batch:
        data, label, frame_id, num_points_in_box, box, center, file_name, points = item
        data_tensor = torch.tensor(data, dtype=torch.float)
        padding_size = max_size - data_tensor.shape[0]
        padded_sample = F.pad(data_tensor, (0, 0, 0, padding_size), 'constant', 0)
        padded_data.append(padded_sample)
        labels.append(label)
        frame_ids.append(frame_id)
        num_points_in_boxes.append(num_points_in_box)
        boxes.append(box)
        centers.append(center)
        file_names.append(file_name)
        points_list.append(points)

    padded_data = torch.stack(padded_data, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    frame_ids = torch.tensor(frame_ids, dtype=torch.long)
    num_points_in_boxes = torch.tensor(num_points_in_boxes, dtype=torch.long)
    boxes = torch.stack(boxes, dim=0)
    centers = np.array(centers)

    return padded_data, labels, frame_ids, num_points_in_boxes, boxes, centers, file_names, points_list

def evaluate(args, test_loader, model, device):
    model.eval()
    results = []
    input_box_count = 0  # Initialize input box counter
    output_box_count = 0  # Initialize output box counter

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            data, label, frame_id, num_points_in_box, boxes, centers, file_names, points_list = batch
            torch.cuda.empty_cache()
            input_box_count += len(boxes)  # Count the number of input boxes

            data = data.permute(0, 2, 1).to(device)
            output = model(data)
            preds = output.max(dim=1)[1]

            for i in range(len(frame_id)):
                if frame_id[i] == 0:
                    frame_id[i] = int(file_names[i].split('_')[1].split('.')[0])  # file_name에서 frame_id 추출
                
                bounding_boxes = boxes[i].cpu().numpy().reshape(-1, 7)
                
                results.append({
                    'frame_id': int(frame_id[i]),
                    'file_name': file_names[i],
                    'label': 'normal' if preds[i].cpu().numpy() == 0 else 'fake',
                    'num_points_in_box': int(num_points_in_box[i].cpu().numpy()),
                    'boxes': bounding_boxes.tolist(),
                    'box_center': centers[i].tolist(),
                    'box_dims': bounding_boxes[0][3:6].tolist() if len(bounding_boxes) > 0 else [0, 0, 0],
                    'box_yaw': bounding_boxes[0][6].tolist() if len(bounding_boxes) > 0 else 0,
                    'points': points_list[i]
                })
                output_box_count += 1  # Count the number of output boxes

                # Stage 2의 frame_id 로그 확인 ####
                print(f"Stage 2 Frame ID: {frame_id[i]}, Pred: {preds[i].cpu().numpy()}, Num Points: {num_points_in_box[i].cpu().numpy()}, Box Center: {centers[i]}, File Name: {file_names[i]}") ####

    return results, input_box_count, output_box_count

def main():
    parser = argparse.ArgumentParser(description='DGCNN for Point Cloud Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the point cloud data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained DGCNN model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per point cloud')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--k', type=int, default=4, help='Number of nearest neighbors to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    args = parser.parse_args()

    test_dataset = BoundingBoxDataset(args.data_path, args.num_points)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(args, output_channels=40).to(device)

    state_dict = torch.load(args.model_path)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    results, input_box_count, output_box_count = evaluate(args, test_loader, model, device)

    with open(os.path.join(args.data_path, 'stage2_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f'Total boxes in Stage 1: {len(test_dataset.data)}')
    print(f'Total input boxes in Stage 2: {input_box_count}')
    print(f'Total output boxes in Stage 2: {output_box_count}')

if __name__ == '__main__':
    main()
