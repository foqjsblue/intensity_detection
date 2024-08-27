import os
import struct
import shutil


def count_points(bin_file):
    with open(bin_file, 'rb') as f:
        content = f.read()
        num_points = len(content) // (4 * 4)  # Each point has x, y, z, intensity (4 floats)
    return num_points


def process_files(input_dirs, output_dirs):
    counts = {'200~399': 0, '400~599': 0, '600~799': 0, '800~999': 0, '1000↑': 0}

    for input_dir in input_dirs:
        for bin_file in os.listdir(input_dir):
            if bin_file.endswith('.bin'):
                file_path = os.path.join(input_dir, bin_file)
                num_points = count_points(file_path)

                if 200 <= num_points < 400:
                    shutil.copy(file_path, output_dirs['200~499'])
                    counts['200~399'] += 1
                elif 400 <= num_points < 600:
                    shutil.copy(file_path, output_dirs['500~999'])
                    counts['400~599'] += 1
                elif 600 <= num_points < 800:
                    shutil.copy(file_path, output_dirs['500~999'])
                    counts['600~799'] += 1
                elif 800 <= num_points < 1000:
                    shutil.copy(file_path, output_dirs['500~999'])
                    counts['800~999'] += 1
                elif num_points >= 1000:
                    shutil.copy(file_path, output_dirs['1000↑'])
                    counts['1000↑'] += 1

    return counts


# Define input and output directories
input_dirs = ['input_folder1', 'input_folder2', 'input_folder3', 'input_folder4', 'input_folder5']
output_dirs = {
    '200~399': 'output_folder_200_399',
    '400~599': 'output_folder_400_599',
    '600~799': 'output_folder_600_799',
    '800~999': 'output_folder_800_999',
    '1000↑': 'output_folder_1000_up'
}

# Create output directories if they don't exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Process files and get the counts
counts = process_files(input_dirs, output_dirs)

# Print the counts
for category, count in counts.items():
    print(f'Number of files in {category}: {count}')
