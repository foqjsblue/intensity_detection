import numpy as np
import os
from pathlib import Path


def modify_intensity(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in input_folder.glob('*.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        points[:, 3] = np.random.uniform(0.27, 0.3922, size=points.shape[0])
        # points[:, 3] = 1.0
        # points[:, 3] *= 3.0

        # Save the modified data to the output folder
        output_file_path = output_folder / file_path.name
        points.tofile(output_file_path)

input_folder = 'kitti_box/box_cyc/testing/box_cyc_test'  # The input folder path containing the .bin files
output_folder = 'kitti_box/box_cyc/testing/box_cyc_test_rd027039'  # The output folder path where the modified .bin files will be saved

modify_intensity(input_folder, output_folder)
