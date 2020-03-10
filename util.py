import pickle
import queue

import logging
import os
import tqdm
import numpy as np
import pandas as pd

from models import *

model_map = {
    'fixed': 'FixedDoseModel',
    'clinical': 'WarfarinClinicalDosingModel',
    'pharmacogenetic': 'WarfarinPharmacogeneticDosingModel'
}

ckpt_paths = queue.Queue()


def discretize(values):
    """
    Discretizes the values based on the following logic:

        0 ( low ): less than 21 mg/week
        1 ( medium ):  21-49 mg/week
        2 ( high ): more than 49 mg/week

    """

    buckets = np.array([21.0, 49.0])
    disc_values = np.digitize(values, buckets)

    return disc_values


def get_model_class(model_name):
    """
    Maps a model name to it's corresponding class

    """
    return globals()[model_map[model_name]]


def save_model(args, step, model, log):
    """
    Save model parameters to disk.

    Args:
        args: command line arguments given to the program
        step (int): Total number of examples seen during subdir so far.
        model (torch.nn.DataParallel): Model to save.
    """

    checkpoint_path = os.path.join(args.save_dir,
                                   f'step_{step}.obj')

    with open(checkpoint_path, "wb") as dump_file:
        model.log = None
        pickle.dump(model, dump_file)
        log.info(f'Saved checkpoint: {checkpoint_path}')
        model.log = log

    ckpt_paths.put(checkpoint_path)

    # Remove a checkpoint if more than max_checkpoints have been saved
    if ckpt_paths.qsize() > args.max_checkpoints:
        old_ckpt = ckpt_paths.get()
        try:
            os.remove(old_ckpt)
            log.info(f'Removed checkpoint: {old_ckpt}')
        except OSError:
            # Avoid crashing if checkpoint has been removed or protected
            pass


def load_model(checkpoint_path):
    """
    Load model parameters from disk.

    Args:
        checkpoint_path (str): Path to checkpoint to load.

    Returns:
        model: Model loaded from checkpoint.
    """

    # Load model object
    with open(checkpoint_path, "rb") as dump_file:
        model = pickle.load(dump_file)

    return model


def read_data_file(file_path, index_col=False):
    """
    Reads the content of a file using pandas

    Args:
        file_path: path to the input file

    Returns:
        a pandas.Dataframe object
    """

    return pd.read_csv(file_path,
                       index_col=index_col,
                       dtype=str,  # Keep all columns as strings
                       keep_default_na=False  # Don't convert blanks/NA to NaN
                       )


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base_dir, name, subdir, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        subdir (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):

        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def isnumber(n):
    """
    Check if a string is a float number

    """

    try:
        float(n)
    except ValueError:
        return False
    return True