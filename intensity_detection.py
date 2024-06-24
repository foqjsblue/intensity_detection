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

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
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
    #R = np.array([[np.cos(box[6]), 0, -np.sin(box[6])],
    #              [np.sin(box[6]), 0, np.cos(box[6])],
    #              [0, 0, 1]])
    rotated_point = np.dot(R.T, point - box[:3])
    return (abs(rotated_point[0]) <= l / 2) and (abs(rotated_point[1]) <= w / 2) and (abs(rotated_point[2]) <= h / 2)


def find_nearest_neighbors(points, k=3):
    # If the size of the point set is less than k, return an empty list
    if len(points) < k:
        return []

    # Convert the list to a NumPy array
    points_array = np.array(points)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points_array[:, :3])
    distances, indices = nbrs.kneighbors(points_array[:, :3])
    return indices

def count_similar_intensity_points(points, neighbors_indices, intensity_threshold=0.01):
    count_knn = 0
    total_intensity = 0

    for i_knn, point in enumerate(points):
        if i_knn < len(neighbors_indices):
            neighbor_idxs = neighbors_indices[i_knn]
            # Select neighbor points using the indices of each neighbor
            neighbors = [points[idx] for idx in neighbor_idxs]

            if all(abs(point[3] - neighbor[3]) <= intensity_threshold for neighbor in neighbors):
                count_knn += 1
                total_intensity += point[3]

    avg_intensity_knn = total_intensity / count_knn if count_knn > 0 else 0
    return count_knn, avg_intensity_knn




def save_points_to_bin(points, output_dir, file_name):

    file_path = os.path.join(output_dir, file_name)

    # Convert the points to a NumPy array and save to a file
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
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    #with torch.no_grad():
        #for idx, data_dict in enumerate(demo_dataset):
            #logger.info(f'Visualized sample index: \t{idx + 1}')
            #data_dict = demo_dataset.collate_batch([data_dict])
            #load_data_to_gpu(data_dict)
            #pred_dicts, _ = model.forward(data_dict)

            #V.draw_scenes(
                #points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)

            #if not OPEN3D_FLAG:
                #mlab.show(stop=True)

    #logger.info('Demo done.')


#####################################################################################################################

    temppointlist = []
    fake_car_count = 0
    real_car_count = 0
    fake_ped_count = 0
    real_ped_count = 0
    fake_cyc_count = 0
    real_cyc_count = 0

    total_bounding_boxes = 0
    fake_car_info = []  # A list to store fake object information
    fake_ped_info = []
    fake_cyc_info = []

    with (((torch.no_grad()))):
        distances = {}
        intensities = {}

        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            bounding_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            class_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            total_bounding_boxes += len(bounding_boxes)

            file_name = demo_dataset.sample_file_list[idx]
            file_name = os.path.basename(file_name)  # Extract only the file name from the full path


            points_in_boxes = [[] for _ in bounding_boxes]
            for p in data_dict['points']:
                x, y, z, i = p[1].item(), p[2].item(), p[3].item(), p[4].item()
                for idx, box in enumerate(bounding_boxes):
                    if is_point_inside_box(np.array([x, y, z]),box) :
                        points_in_boxes[idx].append([x, y, z, i])

            label_idx = class_labels[idx] - 1
            class_name = cfg.CLASS_NAMES[label_idx] if label_idx < len(cfg.CLASS_NAMES) else "UNKNOWN"

            # Save the bounding box
            #output_dir = '../output_box_002741'
            #if not os.path.exists(output_dir):
                #os.makedirs(output_dir)

            #for idx, points_in_box in enumerate(points_in_boxes):

                #if len(points_in_box) >= 10:
                    #file_name = f'box_{class_name}_{idx+1}.bin'
                    #save_points_to_bin(points_in_box, output_dir, file_name)

            for idx, pts in enumerate(points_in_boxes):

                # Initialize the temppointlist array before starting bounding box processing
                temppointlist = []
                temppointlist3 = []
                temppointlist_all = []

                # Create an array containing the coordinates (x, y, z) and intensity (i) of each point
                box_points = np.array([[p[0], p[1], p[2], p[3]] for p in pts if p[3] != 0.0])
                #print(box_points.shape)

                # Create an array by extracting only the intensity of each point
                intensity = np.array([p[3] for p in pts])

                num_points = len(box_points)

                box_center = bounding_boxes[idx][:3]
                distance = math.sqrt(box_center[0] ** 2 + box_center[1] ** 2 + box_center[2] ** 2)

                #label_idx = class_labels[idx] - 1
                #class_name = cfg.CLASS_NAMES[label_idx] if label_idx < len(cfg.CLASS_NAMES) else "UNKNOWN"

                box_points = []
                for p in pts:
                    if p[3] != 0.0:
                        x, y, z, i = p
                        box_points.append([x, y, z, i])

                for p in pts:
                    if p[3] != 0.0:
                        temppointlist.append(p[3])
                    #print(temppointlist)
                    else :
                        temppointlist3.append(p[3])

                for p in pts :
                    temppointlist_all.append(p[3])

                # Calculate and print neighbor points with similar intensities only if the length of temppointlist is 10 or more
                if len(box_points) >= 10:
                    neighbors_indices = find_nearest_neighbors(box_points)
                    count_knn, avg_similar_intensity = count_similar_intensity_points(box_points, neighbors_indices)

                    if count_knn > 0:
                        print(f"▶ Box {idx + 1} : Number of points with at least one neighbor within intensity threshold: {count_knn}, Average Similar Intensity: {avg_similar_intensity:.2f}")
                    else:
                        print(f"▶ Box {idx + 1} : No points with similar intensity")


                zero_num_points = len(temppointlist3)
                temppointlist = sorted(temppointlist, reverse=True)

                if num_points >= 10 :
                    temppointlist2 = sorted(temppointlist, reverse=True)
                    top_percentage = 0.0
                    bottom_percentage = 0.0

                    # Calculate the number of elements corresponding to the top and bottom percentiles
                    num_elements = len(temppointlist2)
                    num_to_remove_top = int(num_elements * top_percentage)
                    num_to_remove_bottom = int(num_elements * bottom_percentage)

                    temppointlist2 = temppointlist2[num_to_remove_top:] if num_to_remove_bottom == 0 else temppointlist2[num_to_remove_top:-num_to_remove_bottom]

                    if len(temppointlist) >= 10:
                        mode_result = stats.mode(temppointlist)
                        mode_intensity = mode_result.mode
                        mode_freq = mode_result.count

                        if isinstance(mode_intensity, np.ndarray) and len(mode_intensity) > 0:
                            mode_intensity_str = f"{mode_intensity[0]:.2f}"
                        else:
                            mode_intensity_str = "N/A"
                    else:
                        mode_intensity_str = "N/A"

                    mode_intensity_str = f"{mode_intensity:.2f}" if mode_intensity is not None else "N/A"

                    print("\n")

                    if len(temppointlist2) > 20:
                        print('Point Intensities:',
                              [f"{val:.2f}" for val in temppointlist2[:10]] + ['...'] + [f"{val:.2f}" for val in temppointlist2[-10:]])
                        print("\n")
                    else:
                        print('Point Intensities:', [f"{val:.2f}" for val in temppointlist2])
                        print("\n")


                    avg_intensity = np.mean(temppointlist_all)
                    std_intensity = np.std(temppointlist2)
                    med_intensity = np.median(temppointlist2)
                    #var_intensity = np.var(temppointlist2)
                    
                    if not class_name in distances:
                        distances[class_name] = []
                        intensities[class_name] = []
                    distances[class_name].append(distance)
                    intensities[class_name].append(avg_intensity)

                    # Calculate skewness and kurtosis
                    box_skewness = skew(intensity)
                    box_kurtosis = kurtosis(intensity)
                    
                    
                    print(f"Box {idx + 1} ({class_name}): Number of all points = {num_points + zero_num_points}, Number of non-zero points = {num_points}, Number of zero-intensity points = {zero_num_points}")
                    
                    mode_intensity_str = f"{mode_intensity:.2f}" if mode_intensity is not None else "N/A"
                    print(f"Average intensity = {avg_intensity:.2f} ({std_intensity:.2f}), Median intensity = {med_intensity:.2f}, Mode intensity = {mode_intensity_str}, Distance: {distance:.2f} meters")
                    #print("\n")
                    
                    print(f"Box {idx + 1} → Skewness = {box_skewness:.2f}, Kurtosis = {box_kurtosis:.2f}")

                    # Convert string values to float
                    temppointlist = [float(i) for i in temppointlist]
                    threshold = 0.3922

                    # Count the number of elements greater than and less than or equal to the threshold
                    count_greater = sum(i > threshold for i in temppointlist_all)
                    count_lesser_or_equal = sum(i <= threshold for i in temppointlist_all)

                    # Calculate the percentage
                    total_elements = len(temppointlist_all)
                    percentage_greater = (count_greater / total_elements) * 100

                    count_greater, count_lesser_or_equal, percentage_greater

                    print(f"Box {idx + 1} → Retro-reflector : {count_greater}, Diffuse reflector : {count_lesser_or_equal}, Ratio of retro-reflector : {percentage_greater:.2f}%")
                    print("\n")

                    # Attack detection
                    if class_name == 'Car' and len(temppointlist) >= 10:
                        
                        if percentage_greater >= 60 :
                            fake_car_count += 1
                            fake_car_info.append((file_name, idx + 1, round(avg_intensity, 2), str(len(temppointlist)) + 'points', str(round(distance, 2)) + 'm'))
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("▷ Fake Car")
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("\n")

                        else:
                            # if class_name == 'Car' and avg_intensity <= 0.4 and percentage_greater <= 25.0:
                            real_car_count += 1

                    else:
                        pass

                    if class_name == 'Pedestrian' and len(temppointlist) >= 10 :

                        if percentage_greater >= 85 :
                            fake_ped_count += 1
                            fake_ped_info.append((file_name, idx + 1, round(avg_intensity, 2), str(len(temppointlist)) + 'points', str(round(distance, 2)) + 'm'))
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("▷ Fake Pedestrian")
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("\n")

                        else:
                            # if class_name == 'Car' and avg_intensity <= 0.4 and percentage_greater <= 25.0:
                            real_ped_count += 1

                    else:
                        pass

                    if class_name == 'Cyclist' and len(temppointlist) >= 10 :

                        if percentage_greater >= 65 :
                            fake_cyc_count += 1
                            fake_cyc_info.append((file_name, idx + 1, round(avg_intensity, 2), str(len(temppointlist)) + 'points', str(round(distance, 2)) + 'm'))
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("▷ Fake Cyclist")
                            print("◇◇◇◇◇◇◇◇◇◇")
                            print("\n")

                        else:
                            # if class_name == 'Car' and avg_intensity <= 0.4 and percentage_greater <= 25.0:
                            real_cyc_count += 1

                    else:
                        pass

                
        print(distances.keys())

    print(f"Total number of bounding boxes: {total_bounding_boxes}")
    print(f"Number of fake cars: {fake_car_count} / {fake_car_count + real_car_count}")
    print("Fake objects detected in files, box numbers:", fake_car_info)
    print(f"Number of fake pedestrians: {fake_ped_count} / {fake_ped_count + real_ped_count}")
    print("Fake pedestrians detected in files, box numbers:", fake_ped_info)
    print(f"Number of fake cyclists: {fake_cyc_count} / {fake_cyc_count + real_cyc_count}")
    print("Fake cyclists detected in files, box numbers:", fake_cyc_info)
    logger.info('Attack Detection Complete.')
    return box_data


if __name__ == '__main__':
    main()
