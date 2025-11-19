"""Create HDF5 file for PE project."""
import argparse
import h5py
import numpy as np
import os
import pickle
import re
import util

from ct import CTPE  # , CONTRAST_HU_MIN, CONTRAST_HU_MAX
from tqdm import tqdm


CONTRAST_HU_MIN = -100
CONTRAST_HU_MAX = 900

def main(args):
    # Match plain "123.npy"
    study_re = re.compile(r'^(\d+)\.npy$')
    study_paths = []
    ctpes = []

    # Read slice-wise labels
    with open(args.slice_list, 'r') as slice_fh:
        slice_lines = [l.strip() for l in slice_fh.readlines() if l.strip()]
    name2slices = {}
    name2info = {}
    for slice_line in slice_lines:
        info, slices = slice_line.split(':')
        slices = slices.strip()
        info = info.split(',')
        studynum, thicc, label, num_slices, phase, dataset = int(info[0]), float(info[1]), int(info[2]), int(info[3]), info[4], info[5]
        name2info[studynum] = [thicc, label, num_slices, phase, dataset]
        if slices:
            name2slices[studynum] = [int(n) for n in slices.split(',')]
        else:
            name2slices[studynum] = []

    # Collect list of paths to studies to convert
    voxel_means = []
    voxel_stds = []
    for base_path, _, file_names in os.walk(args.data_dir):
        npy_names = [f for f in file_names if f.endswith('.npy')]
        for name in npy_names:
            match = study_re.match(name)
            if not match:
                continue

            study_num = int(match.group(1))

            # Pull thickness & metadata from the txt (not the filename)
            thicc, label, num_slices, phase, dataset = name2info.get(study_num, [None]*5)
            if thicc is None:
                continue

            # Filter by --use_thicknesses
            if thicc not in args.use_thicknesses:
                continue

            if study_num in name2slices:
                # Add to list of studies
                full_path = os.path.join(base_path, name)
                study_paths.append(full_path)
                pe_slice_nums = name2slices.get(study_num, [])
                print(thicc, label, phase, num_slices)
                ctpes.append(
                    CTPE(
                        study_num,
                        thicc,  # use txt-provided thickness
                        pe_slice_nums,
                        num_slices,
                        dataset,
                        is_positive=label,
                        phase=phase
                    )
                )

                # (Optional) compute normalization stats later if you want:
                # vol = np.load(full_path)
                # get_mean_std(vol, voxel_means, voxel_stds)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_fh:
        pickle.dump(ctpes, pkl_fh)

    print('Wrote {} studies'.format(len(study_paths)))
    if voxel_means and voxel_stds:
        print('Mean {}'.format(np.mean(voxel_means)))
        print('Std {}'.format(np.mean(voxel_stds)))
    else:
        print('Mean/Std not computed (no volumes loaded for stats).')


def get_mean_std(scan, means, stds):
    scan = scan.astype(np.float32)
    scan = (scan - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
    scan = np.clip(scan, 0., 1.)
    means.append(np.mean(scan))
    stds.append(np.std(scan))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create HDF5 file for PE')

    parser.add_argument('--data_dir', type=str,
                        default='/users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/',
                        help='Base directory for loading 3D volumes.')
    parser.add_argument('--slice_list', type=str,
                        default='/users/edfarber/scratch/penet/test.txt')
    parser.add_argument('--use_thicknesses', default='1.25', type=str,
                        help='Comma-separated list of thicknesses to use.')
    parser.add_argument('--hu_intercept', type=float, default=-1024,
                        help='Intercept for converting from original numpy files to HDF5 (probably -1024).')
    parser.add_argument('--output_dir', type=str,
                        default='/users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/',
                        help='Output directory for HDF5 file and pickle file.')

    args_ = parser.parse_args()
    args_.use_thicknesses = util.args_to_list(args_.use_thicknesses, arg_type=float, allow_empty=False)

    main(args_)

