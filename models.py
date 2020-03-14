import numpy as np
import re

import util

import torch
import torch.nn as nn
import torch.nn.functional as F

column_mapping = {
    'AGE_IN_DECADES': 10,
    'HEIGHT_IN_CM': 11,
    'WEIGHT_IN_KG': 12,
    'VKORC1_A/G': 2602,
    'VKORC1_A/A': 2601,
    'VKORC1_NA': 2604,
    'CYP2C9_*1/*2': 2578,
    'CYP2C9_*1/*3': 2579,
    'CYP2C9_*2/*2': 2582,
    'CYP2C9_*2/*3': 2583,
    'CYP2C9_*3/*3': 2584,
    'CYP2C9_NA': 2585,
    'RACE_ASIAN': 3,
    'RACE_BLACK_OR_AFRICAN_AMERICAN': 4,
    'RACE_NA': 5,
    'ENZYME_INDUCER_STATUS': 2703,
    'AMIODARONE_1': 2541
}

class LinearRewardModel(nn.Module):

    """
    """

    def __init__(self, hidden_size, drop_prob=0.):
        super(LinearRewardModel, self).__init__()

        self.out_size = 1
        self.drop_prob = drop_prob

        self.beta_1 = nn.Linear(hidden_size, self.out_size, bias=True)
        self.beta_2 = nn.Linear(hidden_size, self.out_size, bias=True)
        self.beta_3 = nn.Linear(hidden_size, self.out_size, bias=True)


    def forward(self, input):

        r1 = self.beta_1(input)
        r2 = self.beta_2(input)
        r3 = self.beta_3(input)

        # Apply dropout
        r1 = F.dropout(r1, self.drop_prob, self.training)
        r2 = F.dropout(r2, self.drop_prob, self.training)
        r3 = F.dropout(r3, self.drop_prob, self.training)

        return r1, r2, r3

class DefaultDoseModel:

    def __init__(self):
        pass

    def train(self, data):
        pass

    def compute_arm_index(self, data):
        pass

class FixedDoseModel(DefaultDoseModel):
    """
     Fixed-dose: This approach will assign 35mg/day (medium) dose to all patients.
    """

    def __init__(self, args):
        super(FixedDoseModel, self).__init__()

        self.dosis = args.fixed_dose


    def get_dose(self, input):
        return self.dosis

    def compute_arm_index(self, data):

        arm_indexes = torch.ones((data.size(0), 1), dtype=torch.long, device=data.device)

        return arm_indexes


class WarfarinClinicalDosingModel(DefaultDoseModel):
    """
     Warfarin clinical dosing: This approach will assign a dose as specified by the
     Warfarin clinical dosing algorithm .
    """

    def __init__(self, args):
        super(WarfarinClinicalDosingModel, self).__init__()

        self.dose_params = [
            4.0376,  # bias term
            -0.2546,  # age
            0.0118,  # height
            0.0134,  # weight
            0,  # VKORC1 A/G
            0,  # VKORC1 A/A
            0,  # VKORC1 genotype unknown
            0,  # CYP2C9 *1/*2
            0,  # CYP2C9 *1/*3
            0,  # CYP2C9 *2/*2
            0,  # CYP2C9 *2/*3
            0,  # CYP2C9 *3/*3
            0,  # CYP2C9 genotype unknown
            -0.6752,  # asian race
            0.4060,  # black or african american
            0.0443,  # missing or mixed race
            1.2799,  # enzyme inducer status
            -0.5695  # amiodarone status
        ]

    def compute_arm_index(self, data):


        dose_inputs = torch.cat([
            torch.ones((data.size(0), 1), dtype=torch.float, device=data.device),
            data[:, list(column_mapping.values())]], 1)

        dose_params_tensor = torch.FloatTensor(self.dose_params).unsqueeze(-1)
        dose_params_tensor.to(data.device)

        dose = torch.matmul(dose_inputs, torch.FloatTensor(self.dose_params))

        dose = dose ** 2

        arm_indexes = torch.LongTensor(util.discretize(dose), device=data.device).unsqueeze(-1)

        return arm_indexes

class WarfarinPharmacogeneticDosingModel(DefaultDoseModel):
    """
     Warfarin pharmacogenetic dosing: This approach will assign a dose as specified by
     the Warfarin pharmacogenetic dosing algorithm.
    """

    def __init__(self, args):
        super(WarfarinPharmacogeneticDosingModel, self).__init__()

        self.dose_params = [
            5.6044,  # bias term
            -0.2614, # age
            0.0087,  # height
            0.0128,  # weight
            -0.8677, # VKORC1 A/G
            -1.6974, # VKORC1 A/A
            -0.4854, # VKORC1 genotype unknown
            -0.5211, # CYP2C9 *1/*2
            -0.9357, # CYP2C9 *1/*3
            -1.0616, # CYP2C9 *2/*2
            -1.9206, # CYP2C9 *2/*3
            -2.3312, # CYP2C9 *3/*3
            -0.2188, # CYP2C9 genotype unknown
            -0.1092, # asian race
            -0.2760, # black or african american
            -0.1032, # missing or mixed race
            1.1816,  # enzyme inducer status
            -0.5503  # amiodarone status
        ]

    def compute_arm_index(self, data):
        dose_inputs = torch.cat([
            torch.ones((data.size(0), 1), dtype=torch.float, device=data.device),
            data[:, list(column_mapping.values())]], 1)

        dose_params_tensor = torch.FloatTensor(self.dose_params).unsqueeze(-1)
        dose_params_tensor.to(data.device)

        dose = torch.matmul(dose_inputs, torch.FloatTensor(self.dose_params))

        dose = dose ** 2

        arm_indexes = torch.LongTensor(util.discretize(dose), device=data.device).unsqueeze(-1)

        return arm_indexes


class LinUCBModel(DefaultDoseModel):
    """
     LinUCB: This approach will assign a dose based on the Linear UCB RL algorithm.
    """

    def __init__(self, args):
        super(LinUCBModel, self).__init__()

        self.hidden_size = args.hidden_size

        self.A = []
        self.b = []
        self.AI = []
        self.theta = []

        self.alpha = args.ucb_alpha

        for _ in range(3):
            self.A.append(torch.eye(self.hidden_size).unsqueeze(0))
            self.b.append(torch.zeros((self.hidden_size, 1)).unsqueeze(0))
            self.AI.append(torch.eye(self.hidden_size).unsqueeze(0))
            self.theta.append(torch.zeros((self.hidden_size, 1)).unsqueeze(0))

        self.A = torch.cat(self.A, 0)
        self.b = torch.cat(self.b, 0)
        self.AI = torch.cat(self.AI, 0)
        self.theta = torch.cat(self.theta, 0)

    def train(self, data):

        # need to loop through the batch and make it iterative
        for t in range(data.size(0)):

            a_max = self.compute_arm_index(data[t].unsqueeze(0))

            # get the reward rewards
            r1, r2, r3 = self.reward_model(data[t].unsqueeze(0))

            # get the reward reward for the calculated arm
            r_reward = torch.cat([r1, r2, r3], 1)
            r = r_reward.gather(1, a_max).squeeze(-1)

            # update parameters
            x = data[t].unsqueeze(-1)
            x_t = torch.transpose(x, 0, 1)

            self.A[a_max] += torch.matmul(x, x_t)
            self.b[a_max] += r * x
            self.AI[a_max] = torch.inverse(self.A[a_max])
            self.theta[a_max] = torch.matmul(self.AI[a_max], self.b[a_max])

    def compute_arm_index(self, data):

        x = data.unsqueeze(1).unsqueeze(-1)

        x_t = torch.transpose(x, 2, 3)

        p = torch.matmul(torch.transpose(self.theta, 1, 2), x).squeeze(-1).squeeze(-1) + \
            self.alpha * torch.sqrt(
                torch.matmul(
                    torch.matmul(x_t, self.AI),
                    x
                    ).squeeze(-1).squeeze(-1)
                )


        # select the action with highest bound
        a_max = torch.argmax(p, 1, keepdim=True)

        return a_max