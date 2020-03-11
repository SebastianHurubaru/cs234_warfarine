import random
import sched

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import util
from args import get_train_args
from tensorboardX import SummaryWriter
from json import dumps
import pandas as pd
from tqdm import tqdm
import numpy as np


def evaluate(model, oracle_model):
    # Load the evaluation data
    data = util.read_data_file(args.input_file, index_col=0)
    eval_dataset = data.to_numpy(dtype=float)

    # Get data loader
    dataset = TensorDataset(Tensor(eval_dataset[:, :-3]), Tensor(eval_dataset[:, -3:]))
    eval_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=args.data_shuffle,
                             num_workers=args.num_workers)

    total_regret = 0.
    n_right_decisions = 0.
    n_samples = 0.
    with torch.no_grad(), \
         tqdm(total=len(eval_dataset)) as progress_bar:
        for features, true_r in eval_loader:
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
                                     performance=n_right_decisions.item() / n_samples * 100)

    return {'total_regret': total_regret, 'performance': n_right_decisions.item() / n_samples * 100}


def evaluate_oracle(oracle_model):
    # Load the evaluation data
    data = util.read_data_file(args.input_file, index_col=0)
    eval_dataset = data.to_numpy(dtype=float)

    # Get data loader
    dataset = TensorDataset(Tensor(eval_dataset[:, :-3]), Tensor(eval_dataset[:, -3:]))
    eval_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=args.data_shuffle,
                             num_workers=args.num_workers)
    oracle_model.eval()

    n_right_decisions = 0.
    n_samples = 0.
    with torch.no_grad(), \
         tqdm(total=len(eval_dataset)) as progress_bar:
        for features, true_r in eval_loader:
            # Setup for forward
            batch_size = features.size(0)

            # Forward
            r1, r2, r3 = oracle_model(features)

            n_samples += batch_size

            # Compute performance
            r_pred = torch.cat([r1, r2, r3], 1)

            pred_arms = torch.argmax(r_pred, dim=1)
            true_arms = torch.argmax(true_r, dim=1)

            n_right_decisions += torch.sum(pred_arms == true_arms)

            # Log info
            progress_bar.update(batch_size)

            progress_bar.set_postfix(performance=n_right_decisions.item() / n_samples * 100)

    oracle_model.train()

    return {'performance': n_right_decisions.item() / n_samples * 100}



def train_default(args, train_loader, tbx):
    # Load the checkpoint if given as parameter
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(args.load_path)

    else:
        # Get model
        log.info('Building model...')
        model = util.get_model_class(args.model)(args)

    # Load the oracle model
    oracle_model = util.LinearOracle(hidden_size=args.hidden_size,
                                     drop_prob=args.drop_prob)
    oracle_model = torch.nn.DataParallel(oracle_model, args.gpu_ids)

    oracle_model = util.load_oracle_model(oracle_model, args.oracle_load_path, args.gpu_ids, return_step=False)
    oracle_model.eval()

    model.oracle_model = oracle_model

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = 0
    step = 0
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')

        with torch.no_grad(), \
             tqdm(total=len(train_loader.dataset)) as progress_bar:
            for features, true_r in train_loader:

                batch_size = features.size(0)
                model.train(features)

                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch)

                steps_till_eval -= batch_size
                step += batch_size

                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')

                    results = evaluate(model, oracle_model)

                    util.save_model(args, step, model, log)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'{results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'train/{k}', v, step)


def train_oracle(args, train_loader, tbx):
    # Get model
    log.info('Building model...')

    model = util.LinearOracle(hidden_size=args.hidden_size,
                              drop_prob=args.drop_prob)

    model = torch.nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_oracle_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=False,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = torch.optim.Adadelta(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.l2_wd)


    # scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    scheduler = sched.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
             tqdm(total=len(train_loader.dataset)) as progress_bar:
            for features, true_r in train_loader:

                # Setup for forward
                batch_size = features.size()[0]
                optimizer.zero_grad()

                # Forward
                r1, r2, r3 = model(features)
                r1, r2, r3 = r1.to(device), r2.to(device), r3.to(device)

                # loss_r1 = F.mse_loss(r1, true_r[:, 0])
                # loss_r2 = F.mse_loss(r2, true_r[:, 1])
                # loss_r3 = F.mse_loss(r3, true_r[:, 2])
                #
                # loss_r1_val = loss_r1.item()
                # loss_r2_val = loss_r2.item()
                # loss_r3_val = loss_r3.item()
                #
                # # Backward
                # loss_r1.backward()
                # loss_r2.backward()
                # loss_r3.backward()

                # loss_r = F.mse_loss(r1, true_r[:, 0]) + F.mse_loss(r2, true_r[:, 1]) + F.mse_loss(r3, true_r[:, 2])
                loss_r = F.mse_loss(torch.cat([r1, r2, r3], 1), true_r)
                loss_r_val = loss_r.item()

                # Backward
                loss_r.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                # progress_bar.set_postfix(epoch=epoch,
                #                          mse_loss_r1=loss_r1_val,
                #                          mse_loss_r2=loss_r2_val,
                #                          mse_loss_r3=loss_r3_val)
                #
                # tbx.add_scalar('oracle/mse_loss_r1', loss_r1_val, step)
                # tbx.add_scalar('oracle/mse_loss_r2', loss_r2_val, step)
                # tbx.add_scalar('oracle/mse_loss_r3', loss_r3_val, step)

                progress_bar.set_postfix(epoch=epoch,
                                         mse_loss=loss_r_val)

                tbx.add_scalar('oracle/mse_loss', loss_r_val, step)
                tbx.add_scalar('oracle/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    results = evaluate_oracle(model)
                    # saver.save(step, model, loss_r1_val + loss_r2_val + loss_r3_val, device)
                    saver.save(step, model, loss_r_val, device)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    # loss_str = f'mse_loss_r1 - {loss_r1_val} mse_loss_r2 - {loss_r2_val} mse_loss_r3 - {loss_r3_val}'
                    loss_str = f'mse_loss - {loss_r_val}'
                    log.info(f'Oracle {results_str} {loss_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'oracle/{k}', v, step)


if __name__ == '__main__':

    global log

    args = get_train_args()

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, subdir='train')
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)

    device, args.gpu_ids = util.get_available_devices()

    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the data
    log.info('Loading data...')
    data = util.read_data_file(args.input_file, index_col=0)
    train_dataset = data.to_numpy(dtype=float)

    # Get data loader
    log.info('Building dataset...')
    dataset = TensorDataset(Tensor(train_dataset[:, :-3]), Tensor(train_dataset[:, -3:]))
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=args.data_shuffle,
                              num_workers=args.num_workers)

    if args.model == 'oracle':
        train_oracle(args, train_loader, tbx)
    else:
        train_default(args, train_loader, tbx)

    tbx.close()
