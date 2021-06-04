# External Imports
from torch import tensor
import torch
import torch.nn as nn
import unittest

# Internal Imports 
from alpha import Alpha 
from hierarchical_operation import HierarchicalOperation
from mixed_operation import MixedOperation
from operations import SIMPLE_OPS, LEN_SIMPLE_OPS, Double, Triple, Identity, Zero


class TestHierarchicalOperation(unittest.TestCase):

  def test_1level(self):
    '''
    Testing MNAS with just 1 level.
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
    Testing HierarchicalOperation create_dag with 2 levels.
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

  def test_hierarchial_operation_convergence(self):

    print("Hierarchical Operation")

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

    num_epochs = 1000
    operations = {}
    operations[str((0, 1))] = MixedOperation([Double(1, 1), Triple(1, 1), Zero(1, 1, 1), Identity(1, 1)], nn.Parameter(tensor([0., 0., 0., 0.])))
    model = HierarchicalOperation(2, operations)

    print(list(model.parameters()))
    
    alpha_optim = torch.optim.Adam(
            params=model.parameters(),
            lr=0.1,
            betas=(0.5, 0.999),
            weight_decay=1)

    for param in model.parameters():
      param.register_hook(
          lambda grad: 
            print(grad)
            ) 

    for _ in range(0, num_epochs):
      alpha_optim.zero_grad()
      loss = nn.L1Loss()(model(x), y)
      loss.backward()
      print(loss)
      alpha_optim.step()

    for param in model.parameters():
      print(param)


if __name__ == '__main__':
  unittest.main()