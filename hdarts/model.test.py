import unittest
from torch import tensor, zeros
from alpha import Alpha
from operations import SIMPLE_OPS, LEN_SIMPLE_OPS
from model import MixedOperation, HierarchicalOperation, Model, ModelController

class TestMixedOperation(unittest.TestCase):

  def test_primitives(self):
    '''
    Test with primitive operations directly.
    '''

    x = tensor([[
      [
        [1, 1],
        [1, 1]
      ]
    ]])

    # Initialize primitives
    primitives = []
    primitives.append(SIMPLE_OPS["double"](C=1, stride=1, affine=False))
    primitives.append(SIMPLE_OPS["triple"](C=1, stride=1, affine=False))
    
    alpha_e = zeros(len(primitives))
    mixed_op = MixedOperation(operations=primitives, alpha_e=alpha_e)

    y = tensor([[
      [
        [2.5, 2.5],
        [2.5, 2.5]
      ]
    ]])

    assert(y.equal(mixed_op(x)))


class TestHierarchicalOperation(unittest.TestCase):

  def test_1level(self):
    '''
    Testing hdarts with just 1 level.
    Equivalent to darts in this case. Only Mixed Operations of primitives on nodes.
    Only tests base case of create_dag.
    '''
    x = tensor([
    [
      # feature 1
      [
        [1, 1],
        [1, 1]
      ]
    ]
    ])

    # Initialize Alpha
    alpha = Alpha(1, {0: 3}, {0: LEN_SIMPLE_OPS})

    hierarchical_op = HierarchicalOperation.create_dag(
      level=0, 
      alpha=alpha, 
      alpha_dag=alpha.parameters[0][0],
      primitives=SIMPLE_OPS,
      channels_in=1
    )

    y = tensor([[
      # feature 1
      [
        [1.5, 1.5],
        [1.5, 1.5]
      ],
      # feature 2
      [
        [2.25, 2.25],
        [2.25, 2.25]
      ]
    ]])

    assert(y.equal(hierarchical_op(x)))

  def test_2level(self):
    '''
    Testing hdarts with just 1 level.
    Equivalent to darts in this case. Only Mixed Operations of primitives on nodes.
    Only tests base case of create_dag.
    '''
    x = tensor([
    [
      # feature 1
      [
        [1, 1],
        [1, 1]
      ]
    ]
    ])

    # Initialize Alpha
    alpha = Alpha(2, {0: 3, 1: 3}, {0: LEN_SIMPLE_OPS, 1: 1})

    # Create hierarchical operation 
    hierarchical_op = HierarchicalOperation.create_dag(
      level=1, 
      alpha=alpha, 
      alpha_dag=alpha.parameters[1][0],
      primitives=SIMPLE_OPS,
      channels_in=1
    )

    y = tensor([
    [
      [
        [0.7500, 0.7500],
        [0.7500, 0.7500]
      ],

      [
        [1.1250, 1.1250],
        [1.1250, 1.1250]
      ],

      [
        [0.5625, 0.5625],
        [0.5625, 0.5625]
      ],

      [
        [0.84375, 0.84375],
        [0.84375, 0.84375]
      ],

      [
        [0.84375, 0.84375],
        [0.84375, 0.84375]
      ],

      [
        [1.265625, 1.265625],
        [1.265625, 1.265625]
      ]
    ]
    ])

    assert(y.equal(hierarchical_op(x)))

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

    # For every sample, ensure that the vector returned is a valid probability distribution
    sum = 0
    for sample in model(x.float()):
      for prob in sample: 
        assert(prob >= 0 and prob <= 1)
        sum += prob
      assert(sum == 1)
      

if __name__ == '__main__':
  unittest.main()