# External Imports
import torch.nn as nn

# Internal Imports
from alpha import Alpha
from config import SearchConfig


class LearntModel(nn.module):
    def init(self, alpha_normal: Alpha, alpha_reduce: Alpha, config: SearchConfig):
        