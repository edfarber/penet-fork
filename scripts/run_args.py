import argparse
import json
import subprocess
import sys

def run_args(args):
    # Build command from args
    cmd = [sys.executable, 'train.py']
    cmd += ['--{}={}'.format(k, v) for k, v in args.items()]

    # Spawn process to run the args, letting stdout/stderr flow to the console
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('args_path', type=str, help='Path to JSON file with args to run.')
    script_args = parser.parse_args()

    # Load args from args.json file
    with open(script_args.args_path, 'r') as json_file:
        args_ = json.load(json_file)

    print('Running job with args from {}...'.format(script_args.args_path))
    run_args(args_)

