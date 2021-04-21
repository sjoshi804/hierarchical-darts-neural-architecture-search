# External Imports
import torch.nn as nn

# Internal Imports
import Alpha from alpha
import SearchConfig


class LearntModel(nn.module):
    def init(self, alpha_normal: Alpha, alpha_reduce: Alpha, config: SearchConfig):
        

