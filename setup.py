"""
Pre-process Warfarin data.

Usage:
    > python setup.py

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""
import re

import pandas as pd

import util
from args import get_setup_args


def derive_rs9923231_genotype(race, rs2359612, rs9934438, rs8050894):
    """
    Imputation of VKORC1 SNPs based on Section S4 from appendix

    """

    rs9923231 = 'NA'

    if race is not ('Black or African American' or 'Unknown'):
        if rs2359612 == 'C/C':
            rs9923231 = 'G/G'
        elif rs2359612 == 'T/T':
            rs9923231 = 'A/A'
        elif rs9923231 == 'C/T':
            rs9923231 = 'A/G'

    if rs9934438 == 'C/C': rs9923231 = 'G/G'
    if rs9934438 == 'T/T': rs9923231 = 'A/A'
    if rs9934438 == 'C/T': rs9923231 = 'A/G'

    if race is not ('Black or African American' or 'Unknown'):
        if rs8050894 == 'G/G':
            rs9923231 = 'G/G'
        elif rs8050894 == 'C/C':
            rs9923231 = 'A/A'
        elif rs8050894 == 'C/G':
            rs9923231 = 'A/G'

    return rs9923231


def remove_no_warfare_dosis_records(df):
    """
    Remove records with no Warfarin dose
    
    """
    df = df.drop(df[df['Therapeutic Dose of Warfarin'] == 'NA'].index)

    return df


def pre_process(df):
    # Remove blank data
    df = df.drop(df[df['PharmGKB Subject ID'] == ''].index)

    # Remove unlabeled data
    df = remove_no_warfare_dosis_records(df)

    # Fill the rs9923231 where not available
    df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = \
        df.apply(
            lambda row: derive_rs9923231_genotype(
                row['Race'],
                row['VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G'],
                row['VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G'],
                row['VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G']
                if row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] == 'NA'
                else row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']
            ), axis=1)

    return df


def extract_age_as_decades(row):
    # age in decades: 1 for 10-19, 2 for 20-29
    age_str = re.split('\\+|-', row['Age'])[0].strip()
    age = int(age_str) // 10 if age_str.isnumeric() else 0

    log.debug(
        'ID {} - Age = {} => {}'
            .format(
            row['PharmGKB Subject ID'], row['Age'], age
        )
    )

    return age


def extract_height_in_cm(row):
    height = float(row['Height (cm)']) if util.isnumber(row['Height (cm)']) else 0

    log.debug(
        'ID {} - Height = {} => {}'
            .format(
            row['PharmGKB Subject ID'], row['Height (cm)'], height
        )
    )

    # height in cm
    return height


def extract_weight_in_kg(row):
    # weight in kg
    weight = float(row['Weight (kg)']) if util.isnumber(row['Weight (kg)']) else 0

    log.debug(
        'ID {} - Weight = {} => {}'
            .format(
            row['PharmGKB Subject ID'], row['Weight (kg)'], weight
        )
    )

    return weight


def extract_vkorc1(row, value):

    # VKORC1(rs9923231)
    vkorc1 = 1 if row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] == value else 0

    log.debug(
        'ID {} - VKORC1(rs9923231)_{} = {} => {}'
            .format(
            row['PharmGKB Subject ID'], value,
            row['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'], vkorc1
        )
    )

    return vkorc1


def extract_cyp2c9(row, value):
    # CYP2C9 genotype
    cyp2c9 = 1 if row['Cyp2C9 genotypes'] == value else 0

    log.debug(
        'ID {} - CYP2C9_{} = {} => {}'
            .format(
            row['PharmGKB Subject ID'], value,
            row['Cyp2C9 genotypes'], cyp2c9
        )
    )

    return cyp2c9


def extract_race(row, value):
    race = 1 if row['Race'] == value else 0

    log.debug(
        'ID {} - Race_{} = {} => {}'
            .format(
            row['PharmGKB Subject ID'], value,
            row['Race'], race
        )
    )

    return race


def extract_enzyme_inducer_status(row):

    carbamazepine = 1 if row['Carbamazepine (Tegretol)'].isnumeric() and int(
        row['Carbamazepine (Tegretol)']) == 1 else 0

    phenytoin = 1 if row['Phenytoin (Dilantin)'].isnumeric() and int(row['Phenytoin (Dilantin)']) == 1 else 0

    rifampin_or_rifampicin = 1 if row['Rifampin or Rifampicin'].isnumeric() and int(
        row['Rifampin or Rifampicin']) == 1 else 0

    enzyme_inducer_status = (carbamazepine + phenytoin + rifampin_or_rifampicin) % 3

    log.debug(
        'ID {} - Carbamazepine (Tegretol) = {}|{},  Phenytoin (Dilantin) = {}|{}, Rifampin or Rifampicin = {}|{} => {}'
            .format(
            row['PharmGKB Subject ID'],
            row['Carbamazepine (Tegretol)'], carbamazepine,
            row['Phenytoin (Dilantin)'], phenytoin,
            row['Rifampin or Rifampicin'], rifampin_or_rifampicin,
            enzyme_inducer_status
        )
    )

    return enzyme_inducer_status


def extract_admiodarone_status(row):

    admiodarone_status = 1 if (row['Amiodarone (Cordarone)'].isnumeric() and int(row['Amiodarone (Cordarone)']) == 1) else 0

    log.debug(
        'ID {} - Amiodarone (Cordarone) = {} => {}'
            .format(
            row['PharmGKB Subject ID'],
            row['Amiodarone (Cordarone)'],
            admiodarone_status
        )
    )

    return admiodarone_status


def get_reward_for_arm(row, arm):

    true_arm = util.discretize(float(row['Therapeutic Dose of Warfarin']))

    reward = 0 if arm - 1 == true_arm else -1

    log.debug(
        'ID {} - Arm_{} Therapeutic Dose of Warfarin = {}|{}  => {}'
            .format(
            row['PharmGKB Subject ID'], arm,
            row['Therapeutic Dose of Warfarin'], true_arm,
            reward
        )
    )

    return reward


def extract_features(df):
    features_df = df.copy(deep=True)

    features_df.drop(features_df.columns.difference(['PharmGKB Subject ID']), 1, inplace=True)

    # Age
    features_df['age_decades'] = df.apply(lambda row: extract_age_as_decades(row), axis=1)

    # Height
    features_df['height_cm'] = df.apply(lambda row: extract_height_in_cm(row), axis=1)

    # Weight
    features_df['weight_kg'] = df.apply(lambda row: extract_weight_in_kg(row), axis=1)

    # VKORC1(rs9923231) A/G - 1
    features_df['VKORC1_A/G'] = df.apply(lambda row: extract_vkorc1(row, 'A/G'), axis=1)

    # VKORC1(rs9923231) A/A - 1
    features_df['VKORC1_A/A'] = df.apply(lambda row: extract_vkorc1(row, 'A/A'), axis=1)

    # VKORC1(rs9923231) NA - 1
    features_df['VKORC1_NA'] = df.apply(lambda row: extract_vkorc1(row, 'NA'), axis=1)

    # CYP2C9 genotype is *1/*2
    features_df['CYP2C9_*1/*2'] = df.apply(lambda row: extract_cyp2c9(row, '*1/*2'), axis=1)

    # CYP2C9 genotype is *1/*3
    features_df['CYP2C9_*1/*3'] = df.apply(lambda row: extract_cyp2c9(row, '*1/*3'), axis=1)

    # CYP2C9 genotype is *2/*2
    features_df['CYP2C9_*2/*2'] = df.apply(lambda row: extract_cyp2c9(row, '*2/*2'), axis=1)

    # CYP2C9 genotype is *2/*3
    features_df['CYP2C9_*2/*3'] = df.apply(lambda row: extract_cyp2c9(row, '*2/*3'), axis=1)

    # CYP2C9 genotype is *3/*3
    features_df['CYP2C9_*3/*3'] = df.apply(lambda row: extract_cyp2c9(row, '*3/*3'), axis=1)

    # CYP2C9 genotype is NA
    features_df['CYP2C9_NA'] = df.apply(lambda row: extract_cyp2c9(row, 'NA'), axis=1)

    # asian race
    features_df['race_asian'] = df.apply(lambda row: extract_race(row, 'Asian'), axis=1)

    # black or african american race
    features_df['race_black_or_african_american'] = df.apply(lambda row: extract_race(row, 'Black or African American'),
                                                             axis=1)

    # missing or mixed race
    features_df['race_unknown'] = df.apply(lambda row: extract_race(row, 'Unknown'), axis=1)

    # enzyme inducer status: 1 if patient taking carbamazepine, phenytoin, rifampin, or rifampicin, otherwise zero
    features_df['enzyme_inducer_status'] = df.apply(lambda row: extract_enzyme_inducer_status(row), axis=1)

    # admiodarone status, i.e. patient taking amiodarone
    features_df['admiodarone_status'] = df.apply(lambda row: extract_admiodarone_status(row), axis=1)

    # get the rewards to be used as labels
    features_df['reward_arm_1'] = df.apply(lambda row: get_reward_for_arm(row, 1), axis=1)

    # get the rewards to be used as labels
    features_df['reward_arm_2'] = df.apply(lambda row: get_reward_for_arm(row, 2), axis=1)

    # get the rewards to be used as labels
    features_df['reward_arm_3'] = df.apply(lambda row: get_reward_for_arm(row, 3), axis=1)

    return features_df


if __name__ == '__main__':
    global log

    # Get command-line args
    args = get_setup_args()
    args.name = 'setup'

    log = util.get_logger(util.get_save_dir(args.save_dir, args.name, subdir=args.name), args.name)

    # Read the input file
    df = util.read_data_file(args.orig_input_file)

    # Pre-process the data
    df = pre_process(df)

    # Save the pre-processed data
    df.to_csv(args.out_file, index=False)

    # Extract features
    feat_df = extract_features(df)

    # Save features
    feat_df.to_csv(args.out_feat_file, index=False)
