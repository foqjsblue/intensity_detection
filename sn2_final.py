import subprocess
import argparse

def run_save_box_fake(data_path, cfg_file, ckpt, ext='.bin'):
    cmd = [
        'python', 'sn2_1stage.py',
        '--data_path', data_path,
        '--cfg_file', cfg_file,
        '--ckpt', ckpt,
        '--ext', ext
    ]
    subprocess.run(cmd)

def run_main(data_path, model_path):
    cmd = [
        'python', 'sn2_2stage.py',
        '--data_path', data_path,
        '--model_path', model_path
    ]
    subprocess.run(cmd)

def run_filter_fake_points(data_path, output_path):
    cmd = [
        'python', 'sn2_filter.py',
        '--data_path', data_path,
        '--output_path', output_path
    ]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Run Point Cloud Detection')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the point cloud data')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to the config file for sn2_1stage.py')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file for sn2_1stage.py')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model for sn2_2stage.py')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the filtered point cloud data')
    args = parser.parse_args()

    # Run the first stage: sn2_1stage.py
    run_save_box_fake(args.data_path, args.cfg_file, args.ckpt)

    # Run the second stage: sn2_2stage.py
    run_main(args.data_path, args.model_path)

    # Filter out fake points and save the results
    run_filter_fake_points(args.data_path, args.output_path)

if __name__ == '__main__':
    main()
