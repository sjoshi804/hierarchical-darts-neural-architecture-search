# External Imports
from operations import  SIMPLE_OPS, LEN_SIMPLE_OPS
from torch import tensor
import unittest

# Internal Imports
from alpha import Alpha
from model import Model


class TestModel(unittest.TestCase):
  def test_2level_model(self):
    x = tensor([
    [
      # feature 1
      [
        [1., 1.],
        [1., 1.]
      ]
    ]
    ])

    # Initialize Alpha
    alpha = Alpha(2, {0: 3, 1: 3}, {0: LEN_SIMPLE_OPS, 1: 1})

    model = Model(
      alpha=alpha,
      primitives=SIMPLE_OPS,
      channels_in=1,
      channels_start=2,
      stem_multiplier=1,
      num_classes=5)

    raise NotImplementedError

if __name__ == '__main__':
  unittest.main()