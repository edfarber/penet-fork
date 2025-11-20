import sys
import os
import json
import subprocess
import argparse

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from args import TestArgParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to train_configs.json')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--phase', type=str, default='val', help='Phase to test on (val/test)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU IDs to use')
    
    args = parser.parse_args()
    
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    # Get valid test args from TestArgParser
    test_parser = TestArgParser()
    # Access the underlying argparse object
    valid_args = set(action.dest for action in test_parser.parser._actions)
    
    # Build command
    cmd = [sys.executable, 'test.py']
    
    # Add config args if they are valid test args
    print("Adding valid arguments from config:")
    for k, v in config.items():
        if k in valid_args:
            # Skip args we want to override or handle specially
            if k in ['ckpt_path', 'data_dir', 'phase', 'gpu_ids', 'results_dir']:
                continue
            
            # Handle list arguments (like pe_types)
            if isinstance(v, list):
                # argparse expects lists as string representation or multiple args?
                # BaseArgParser uses type=eval for pe_types, so it expects a string representation of the list
                # e.g. "['central', 'segmental']"
                # So we should pass the string representation
                val_str = str(v).replace("'", '"') # Use double quotes for JSON compatibility if needed, but python eval handles single quotes too.
                # However, passing it via subprocess, we need to be careful with quotes.
                # If we pass it as a single argument string, subprocess handles escaping.
                cmd.append(f'--{k}={val_str}')
            else:
                cmd.append(f'--{k}={v}')
            # print(f"  --{k}={v}")
            
    # Add overrides
    cmd.append(f'--ckpt_path={args.ckpt_path}')
    cmd.append(f'--data_dir={args.data_dir}')
    cmd.append(f'--phase={args.phase}')
    
    # Handle pkl_path default logic
    pkl_path = config.get('pkl_path', '')
    if not pkl_path:
        pkl_path = os.path.join(args.data_dir, 'series_list.pkl')
        print(f"pkl_path not specified in config, defaulting to: {pkl_path}")
    cmd.append(f'--pkl_path={pkl_path}')
    
    # Handle GPU IDs
    if args.gpu_ids is not None:
        cmd.append(f'--gpu_ids={args.gpu_ids}')
    elif 'gpu_ids' in config:
        cmd.append(f'--gpu_ids={config["gpu_ids"]}')
        
    print(f"\nRunning command:\n{' '.join(cmd)}\n")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
