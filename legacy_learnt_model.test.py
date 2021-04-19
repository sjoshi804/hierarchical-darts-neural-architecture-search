# External Imports
from torch import tensor
import torch.nn as nn
import unittest

# Internal Imports 
from alpha import Alpha 
from learnt_model import LearntModel
from model import Model
from operations import SIMPLE_OPS, LEN_SIMPLE_OPS

class learntModelTest(unittest.TestCase):
    def test_mixed_operation_finalization(self):
        raise NotImplementedError 

    def test_hierarchical_operation_finalization(self):
        raise NotImplementedError

    def test_model_finalization(self):
        # Initialize Alpha
        alpha = Alpha(2, {0: 2, 1: 2}, {0: LEN_SIMPLE_OPS, 1: 1})

        # Set alpha_e for edge of op on ground level 
        alpha.parameters[0][0][(0,1)] = nn.Parameter(tensor([10., 0., 0., 0.]))
        alpha.parameters[1][0][(0,1)] = nn.Parameter(tensor([10., 0.]))

        #Create simple model
        model = Model(
        alpha=alpha,
        primitives=SIMPLE_OPS,
        channels_in=1,
        channels_start=2,
        stem_multiplier=1,
        num_classes=5,
        test_mode=True)

        # Input
        x = tensor([
        [
        # feature 1
        [
            [1.]
        ]
        ]
        ])

        # Expected output
        y = tensor([
        [
        # feature 1
        [
            [2.]
        ]
        ]
        ])

        learnt_model = LearntModel(model)
        assert(learnt_model(x).equal(y))



if __name__ == '__main__':
  unittest.main()