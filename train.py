import util
from args import get_train_args
from tensorboardX import SummaryWriter
from json import dumps
import pandas as pd
from tqdm import tqdm
import numpy as np


def evaluate(model, data):

    pred_val = util.discretize(model.evaluate(data.to_numpy(dtype=float)[:, :-1]))
    true_val = util.discretize(data['true_dose'].to_numpy(dtype=float))

    performance = np.sum((pred_val - true_val) == 0)/len(true_val)

    return performance


def train_default(args, tbx):

    # Load the checkpoint if given as parameter
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(args.load_path)

    else:
        # Get model
        log.info('Building model...')
        model = util.get_model_class(args.model)()

    # Load the data
    log.info('Loading data...')
    data = util.read_data_file(args.input_file, index_col=0)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = 0
    step = 0
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')

        with tqdm(total=len(data)) as progress_bar:
            for index, row in data.iterrows():

                # remove the true dose from the end of the array
                model.train(row.to_numpy(dtype=float)[:-1])

                progress_bar.update(1)
                progress_bar.set_postfix(epoch=epoch)

                steps_till_eval -= 1
                step += 1

                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    performance = evaluate(model, data)

                    util.save_model(args, step, model, log)

                    # Log to console
                    results_str = 'Performance: {}'.format(performance)
                    log.info(f'{results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    tbx.add_scalar(f'train/perf', performance * 100, step)


def train_oracle(args, tbx):
    pass


if __name__ == '__main__':

    global log

    args = get_train_args()

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, subdir='train')
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)

    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    if args.model == 'oracle':
        train_oracle(args, tbx)
    else:
        train_default(args, tbx)

    tbx.close()