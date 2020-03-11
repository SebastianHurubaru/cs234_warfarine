import util
from args import get_train_args, get_test_args
from tensorboardX import SummaryWriter
from json import dumps
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


if __name__ == '__main__':

    args = get_test_args()

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, subdir='test')
    log = util.get_logger(args.save_dir, args.name)
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Load the checkpoint if given as parameter
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(args.load_path)

    else:
        # Get model
        log.info('Building model...')
        model = util.get_model_class(args.model)()

    # Load the oracle model
    oracle_model = util.LinearOracle(hidden_size=args.hidden_size,
                                     drop_prob=args.drop_prob)
    oracle_model = torch.nn.DataParallel(oracle_model, gpu_ids)

    oracle_model = util.load_oracle_model(oracle_model, args.oracle_load_path, gpu_ids, return_step=False)
    oracle_model.eval()

    # Load the data
    log.info('Loading data...')

    data = util.read_data_file(args.input_file, index_col=0)
    test_dataset = data.to_numpy(dtype=float)

    # Get data loader
    log.info('Building dataset...')
    dataset = TensorDataset(Tensor(test_dataset[:, :-3]), Tensor(test_dataset[:, -3:]))
    test_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=args.data_shuffle,
                             num_workers=args.num_workers)

    # Evaluating
    log.info('Evaluating...')

    total_regret = 0.
    n_right_decisions = 0.
    n_samples = 0.
    with torch.no_grad(), \
         tqdm(total=len(dataset)) as progress_bar:

        for features, true_r in test_loader:
            # Setup for forward
            batch_size = features.size(0)

            # Forward
            r1, r2, r3 = oracle_model(features)

            # Compute regret
            r = torch.cat([r1, r2, r3], 1)
            r_max, best_arms = torch.max(r, dim=1)
            pred_arms = model.compute_arm_index(features)
            r_pred = r.gather(1, pred_arms).squeeze(-1)

            total_regret += torch.sum(r_max - r_pred)

            # Compute performance
            n_samples += batch_size
            pred_arms = model.compute_arm_index(features).squeeze(-1)
            true_arms = torch.argmax(true_r, dim=1)
            n_right_decisions += torch.sum(pred_arms == true_arms)


            # Log info
            progress_bar.update(batch_size)

            progress_bar.set_postfix(total_regret=total_regret.item(),
                                     performance=n_right_decisions.item()/n_samples * 100)

    log.info(f'Model - {args.model} total_regret = {total_regret.item()}')
    log.info(f'Model - {args.model} performance = {n_right_decisions.item()/n_samples*100}')




