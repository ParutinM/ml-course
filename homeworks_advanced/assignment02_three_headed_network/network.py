
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title = nn.Sequential(
            nn.Embedding(n_tokens, embedding_dim=hid_size),
            Reorder(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            # nn.BatchNorm1d(hid_size)
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            Flatten()
        )
        self.full = nn.Sequential(
            nn.Embedding(n_tokens, embedding_dim=hid_size),
            Reorder(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            # nn.BatchNorm1d(hid_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            Flatten()
        )
        self.category = nn.Sequential(
            nn.Linear(n_cat_features, 4 * hid_size),
            nn.ReLU(),
            nn.Linear(4 * hid_size, 2 * hid_size),
            nn.BatchNorm1d(2 * hid_size),
            nn.ReLU(),
            Flatten()
        )

        # Example for the final layers (after the concatenation)
        self.dense = nn.Sequential(
            nn.Linear(in_features=concat_number_of_features, out_features=4 * hid_size),
            nn.ReLU(),
            nn.Linear(4 * hid_size, 2 * hid_size),
            nn.ReLU(),
            nn.Linear(in_features=hid_size*2, out_features=1)
        )

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title = self.title(input1)
        full = self.full(input2)
        category = self.category(input3)
        
        
        concatenated = torch.cat([title, full, category], dim=1)
        
        out = self.dense(concatenated)
        
        return out