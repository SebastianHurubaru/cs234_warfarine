import util
from args import get_train_args, get_test_args
from tensorboardX import SummaryWriter
from json import dumps
import pandas as pd
from tqdm import tqdm
import numpy as np


def evaluate(model, data):

    pred_val = util.discretize(model.evaluate(data).to_numpy())
    true_val = util.discretize(data['Therapeutic Dose of Warfarin'].to_numpy(dtype=float))

    performance = np.sum((pred_val - true_val) == 0)/len(true_val)

    return performance

if __name__ == '__main__':

    args = get_test_args()

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, subdir=False)
    log = util.get_logger(args.save_dir, args.name)

    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Load the checkpoint if given as parameter
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(args.load_path)

    else:
        # Get model
        log.info('Building model...')
        model = util.get_model_class(args.model)(log)

    # Load the data
    log.info('Loading data...')
    data = util.read_data_file(args.input_file, index_col=0)

    # Evaluating
    log.info('Evaluating...')

    performance = evaluate(model, data)

    # Log to console
    results_str = 'Performance: {}'.format(performance)
    log.info(f'{results_str}')
