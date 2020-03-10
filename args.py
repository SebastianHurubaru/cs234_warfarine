"""
Command-line arguments for setup.py, train.py, test.py.

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import argparse


def get_setup_args():

    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Pre-process Warfarin data')

    add_common_args(parser)

    parser.add_argument('--orig_input_file',
                        type=str,
                        default='./data/orig/warfarin.csv')

    parser.add_argument('--out_file',
                        type=str,
                        default='./data/warfarin.csv',
                        help='Path to the pre-processed output file.')

    parser.add_argument('--out_feat_file',
                        type=str,
                        default='./data/warfarin_feat.csv',
                        help='Path to the pre-processed features only output file.')

    args = parser.parse_args()

    return args


def get_train_args():
    """
    Get arguments needed in train.py.
    """
    parser = argparse.ArgumentParser('Train a model on Warfarin')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for which to train.')

    parser.add_argument('--eval_steps',
                        type=int,
                        default=100,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    args = parser.parse_args()

    return args


def get_test_args():
    """
    Get arguments needed in test.py.
    """
    parser = argparse.ArgumentParser('Test a trained model on Warfarin')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path and args.model not in ['fixed', 'clinical', 'pharmacogenetic']:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):

    """
    Add arguments common to all 3 scripts: setup.py, train.py, test.py
    """

    parser.add_argument('--input_file',
                        type=str,
                        default='./data/warfarin_feat.csv')
    parser.add_argument('--model',
                        type=str,
                        choices=['oracle', 'fixed', 'clinical', 'pharmacogenetic'],
                        default='fixed')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

def add_train_test_args(parser):
    """
    Add arguments common to train.py and test.py
    """
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify subdir or test run.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
