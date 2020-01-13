# -*- coding: utf-8 -*-

"""
Bone Age Assessment Network (BoNet) PyTorch implementation.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Local imports
from .backbone.inception_v3 import InceptionV3

class BoNet(nn.Module):

    def __init__(self, transform_input=False):
        super(BoNet, self).__init__()
        
        # Backbone
        self.inception_v3 = InceptionV3()

        # Gender
        self.gender = DenseLayer(1, 32)
        self.fc_1 = DenseLayer(100384, 1000)
        self.fc_2 = DenseLayer(1000, 1000)
        self.fc_3 = nn.Linear(1000, 1)

    def forward(self, x, y):
        x = self.inception_v3(x)
        y = self.gender(y)
        x = self.fc_1(torch.cat([x, y], 1))
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True)
