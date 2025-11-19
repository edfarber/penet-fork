import argparse
import optuna
import os
import sys
import torch

# Ensure root path for relative imports when executed from scripts directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train import train  # train() now returns final metrics
from args import TrainArgParser

# Hyperparameter search space definitions
OPTIMIZERS = ['sgd', 'adam']  # sgd excluded for early search efficiency
SCHEDULERS = ['cosine_warmup', 'plateau', 'multi_step']


def build_trial_arg_list(base_args, trial):
    """Construct a list of CLI-style arguments for TrainArgParser from sampled trial params."""
    sampled = {}

    sampled['optimizer'] = trial.suggest_categorical('optimizer', OPTIMIZERS)
    if sampled['optimizer'] in ('adam', 'sgd'):
        sampled['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
    else:
        sampled['learning_rate'] = trial.suggest_float('learning_rate', 5e-4, 5e-2, log=True)

    sampled['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    sampled['lr_scheduler'] = trial.suggest_categorical('lr_scheduler', SCHEDULERS)
    sampled['lr_warmup_steps'] = trial.suggest_int('lr_warmup_steps', 1000, 8000, step=1000)
    sampled['lr_decay_gamma'] = trial.suggest_float('lr_decay_gamma', 0.05, 0.5)
    sampled['lr_decay_step'] = trial.suggest_int('lr_decay_step', 40000, 120000, step=40000)

    if sampled['lr_scheduler'] == 'multi_step':
        # Convert milestones to comma separated list
        m1 = trial.suggest_int('milestone1', 30, 80)
        m2 = trial.suggest_int('milestone2', m1 + 10, 150)
        sampled['lr_milestones'] = f"{m1},{m2}"  # TrainArgParser will parse to list
    else:
        sampled['lr_milestones'] = '50,125,250'  # unused for other schedulers

    sampled['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 4.0)
    sampled['dropout_prob'] = trial.suggest_float('dropout_prob', 0.0, 0.4)
    sampled['abnormal_prob'] = trial.suggest_float('abnormal_prob', 0.3, 0.7)
    sampled['batch_size'] = trial.suggest_categorical('batch_size', [4, 6, 8])
    sampled['iters_per_print'] = sampled['batch_size'] * 4
    sampled['iters_per_visual'] = sampled['batch_size'] * 20
    sampled['fine_tuning_lr'] = trial.suggest_float('fine_tuning_lr', 0.0, 1e-4)
    sampled['use_hem'] = str(trial.suggest_categorical('use_hem', [True, False]))

    # Augmentations
    sampled['do_hflip'] = str(trial.suggest_categorical('do_hflip', [True, False]))
    sampled['do_rotate'] = str(trial.suggest_categorical('do_rotate', [True, False]))
    sampled['do_jitter'] = str(trial.suggest_categorical('do_jitter', [True, False]))

    arg_list = [
        '--data_dir', base_args.data_dir,
        '--dataset', base_args.dataset,
        '--name', f"{base_args.name}_trial{trial.number}",
        '--model', base_args.model,
        '--model_depth', str(base_args.model_depth),
        '--num_epochs', str(base_args.search_epochs),
        '--epochs_per_eval', str(base_args.epochs_per_eval),
        '--epochs_per_save', str(base_args.epochs_per_save),
        '--do_classify', 'True',
        '--best_ckpt_metric', base_args.best_ckpt_metric,
        '--gpu_ids', base_args.gpu_ids,
        '--resize_shape', base_args.resize_shape,
        '--crop_shape', base_args.crop_shape,
        '--num_slices', str(base_args.num_slices),
        '--num_workers', str(base_args.num_workers),
        '--fine_tune', 'True',
        '--fine_tuning_boundary', base_args.fine_tuning_boundary,
        '--save_dir', base_args.save_dir,
        '--include_normals', str(base_args.include_normals),
    ]

    # Inject sampled params
    for k, v in sampled.items():
        arg_list.extend([f"--{k}", str(v)])

    return arg_list


def train_one_trial(base_args, trial):
    trial_arg_list = build_trial_arg_list(base_args, trial)
    parser = TrainArgParser()
    args = parser.parser.parse_args(trial_arg_list)  # Use underlying argparse to avoid double saving defaults
    # Complete parse adjustments from BaseArgParser.parse_args
    # Manually invoke logic from BaseArgParser.parse_args (we can't call parse_args() directly since we used parser.parser)
    # Reusing code: simplest is to recreate full CLI style and call parse_args() on TrainArgParser
    # Instead, call higher-level method
    parser = TrainArgParser()
    sys.argv = ['hparam_search'] + trial_arg_list
    args = parser.parse_args()

    metrics = train(args)
    # Choose optimization target
    target_metric_name = 'val_AUROC' if 'val_AUROC' in metrics else 'val_loss'
    target_val = metrics[target_metric_name]

    # Pruning placeholder (can add intermediate metrics if train() returns them)
    return target_val


def main():
    ap = argparse.ArgumentParser(description='Optuna hyperparameter search for PENet')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--dataset', choices=['pe', 'kinetics'], required=True)
    ap.add_argument('--name', required=True, help='Base experiment name prefix')
    ap.add_argument('--model', default='PENetClassifier', choices=['PENet', 'PENetClassifier'])
    ap.add_argument('--model_depth', type=int, default=50)
    ap.add_argument('--gpu_ids', default='0')
    ap.add_argument('--save_dir', default='../hparam_ckpts')
    ap.add_argument('--num_slices', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--fine_tuning_boundary', default='encoders.3')
    ap.add_argument('--include_normals', type=str, default='False')
    ap.add_argument('--resize_shape', default='224,224')
    ap.add_argument('--crop_shape', default='208,208')
    ap.add_argument('--best_ckpt_metric', default='val_AUROC', choices=['val_AUROC', 'val_loss'])
    ap.add_argument('--epochs_per_eval', type=int, default=1)
    ap.add_argument('--epochs_per_save', type=int, default=5)
    ap.add_argument('--search_epochs', type=int, default=40, help='Number of epochs per trial')
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--study_name', default='penet_optuna')
    ap.add_argument('--storage', default=None, help='Optuna storage URI (e.g., sqlite:///penet.db)')
    ap.add_argument('--direction', choices=['maximize', 'minimize'], default='maximize', help='Direction for study metric')

    base_args = ap.parse_args()
    os.makedirs(base_args.save_dir, exist_ok=True)

    def objective(trial):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        return train_one_trial(base_args, trial)

    study = optuna.create_study(
        study_name=base_args.study_name,
        direction=base_args.direction,
        storage=base_args.storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=base_args.trials, show_progress_bar=True)

    print('Best trial:')
    best = study.best_trial
    print(f'Trial #{best.number}, value={best.value}')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Write summary
    summary_path = os.path.join(base_args.save_dir, f'{base_args.study_name}_summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write(f'Best value: {best.value}\n')
        for k, v in best.params.items():
            fh.write(f'{k}: {v}\n')
    print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()
