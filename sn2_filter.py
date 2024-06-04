import os
import pickle
import numpy as np
import glob
import argparse

def is_point_inside_box(point, box):
    box_center = box['box_center']
    box_dims = box['box_dims']
    box_yaw = box['box_yaw']
    l, w, h = box_dims

    # Translate point to the box coordinate system
    translated_point = point[:3] - box_center

    # Rotation matrix for yaw angle
    R = np.array([[np.cos(box_yaw), -np.sin(box_yaw), 0],
                  [np.sin(box_yaw), np.cos(box_yaw), 0],
                  [0, 0, 1]])

    # Rotate the point to align with box's local coordinate system
    rotated_point = np.dot(R, translated_point)

    # Check if the point is inside the box
    is_inside = (abs(rotated_point[0]) <= l / 2) and (abs(rotated_point[1]) <= w / 2) and (abs(rotated_point[2]) <= h / 2)
    
    return is_inside

def filter_fake_points(data_path, output_path):
    stage1_results_path = os.path.join(data_path, 'stage1_results.pkl')
    stage2_results_path = os.path.join(data_path, 'stage2_results.pkl')

    if not os.path.exists(stage1_results_path):
        raise FileNotFoundError(f"Stage1 results file not found: {stage1_results_path}")
    if not os.path.exists(stage2_results_path):
        raise FileNotFoundError(f"Stage2 results file not found: {stage2_results_path}")

    with open(stage1_results_path, 'rb') as f:
        stage1_results = pickle.load(f)

    with open(stage2_results_path, 'rb') as f:
        stage2_results = pickle.load(f)

    fake_boxes = [res for res in stage1_results if res['label'] == 'fake']
    fake_boxes.extend([res for res in stage2_results if res['label'] == 'fake'])

    original_data_files = sorted(glob.glob(os.path.join(data_path, 'merged_*.bin')))
    fake_points_count = 0  # Initialize the counter for fake points
    box_removal_count = {}  # Dictionary to store the count of removed boxes per file

    for data_file in original_data_files:
        points = np.fromfile(data_file, dtype=np.float32).reshape(-1, 4)
        frame_id = int(os.path.splitext(os.path.basename(data_file))[0].split('_')[1])

        fake_boxes_for_frame = [box for box in fake_boxes if box['frame_id'] == frame_id]
        
        filtered_points = []
        removed_points = []  # List to store removed points for debugging
        removed_boxes = 0  # Counter for removed boxes
        
        for point in points:
            is_fake = False
            for box in fake_boxes_for_frame:
                if is_point_inside_box(point, box):
                    is_fake = True
                    fake_points_count += 1
                    removed_points.append(point)
                    removed_boxes += 1
                    break
            if not is_fake:
                filtered_points.append(point)

        filtered_points = np.array(filtered_points, dtype=np.float32)
        
        # Save to output directory instead of overwriting
        filtered_file_path = os.path.join(output_path, os.path.basename(data_file))
        os.makedirs(os.path.dirname(filtered_file_path), exist_ok=True)
        filtered_points.tofile(filtered_file_path)

        # Save the count of removed boxes for this file
        box_removal_count[os.path.basename(data_file)] = removed_boxes

        # Debugging information
        print(f'Filtered points saved to {filtered_file_path}')
        print(f'Number of points before filtering: {len(points)}')
        print(f'Number of points after filtering: {len(filtered_points)}')
        print(f'Number of points removed: {len(removed_points)}')
        print(f'Number of bounding boxes removed in {os.path.basename(data_file)}: {removed_boxes}')
        #print(f'Removed points: {removed_points}')  # Print the removed points for debugging

    print(f'Total fake boxes processed: {len(fake_boxes)}')
    print(f'Total fake points removed: {fake_points_count}')  # Print the number of fake points removed

    # Print the count of removed boxes per file
    for file_name, count in box_removal_count.items():
        print(f'File: {file_name}, Bounding boxes removed: {count}')

def main():
    parser = argparse.ArgumentParser(description='Filter fake points')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the point cloud data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the filtered point cloud data')
    args = parser.parse_args()

    filter_fake_points(args.data_path, args.output_path)

if __name__ == '__main__':
    main()
