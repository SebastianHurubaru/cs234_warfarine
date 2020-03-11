import numpy as np
import re

import util

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearOracle(nn.Module):

    """
    """

    def __init__(self, hidden_size, drop_prob=0.):
        super(LinearOracle, self).__init__()

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

    def __init__(self):
        super(FixedDoseModel, self).__init__()

        self.dosis = 35.0


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

    def __init__(self):
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

    def evaluate(self, eval_data):

        dose_inputs = np.insert(eval_data, 0, 1, axis=1)

        results = np.dot(dose_inputs, np.asarray(self.dose_params).reshape(-1, 1)).reshape(-1)

        results = results ** 2

        return results

    def compute_arm_index(self, data):


        dose_inputs = torch.cat([
            torch.ones((data.size(0), 1), dtype=torch.float, device=data.device),
            data], 1)

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

    def __init__(self):
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

    def evaluate(self, eval_data):

        dose_inputs = np.insert(eval_data, 0, 1, axis=1)

        results = np.dot(dose_inputs, np.asarray(self.dose_params).reshape(-1, 1)).reshape(-1)

        results = results ** 2

        return results

    def compute_arm_index(self, data):


        dose_inputs = torch.cat([
            torch.ones((data.size(0), 1), dtype=torch.float, device=data.device),
            data], 1)

        dose_params_tensor = torch.FloatTensor(self.dose_params).unsqueeze(-1)
        dose_params_tensor.to(data.device)

        dose = torch.matmul(dose_inputs, torch.FloatTensor(self.dose_params))

        dose = dose ** 2

        arm_indexes = torch.LongTensor(util.discretize(dose), device=data.device).unsqueeze(-1)

        return arm_indexes
