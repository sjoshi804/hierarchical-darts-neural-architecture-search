# External Imports
from torch import tensor, zeros
import torch
import torch.nn as nn
import unittest

# Internal Imports
from mixed_operation import MixedOperation
from operations import SIMPLE_OPS, Double, Triple, Zero

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

def test_convergence(self):
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

    num_epochs = 100
    model = MixedOperation([Zero(1, 1, 1), Double(1, 1)], nn.Parameter(tensor([0., 0.])))

    alpha_optim = torch.optim.Adam(
                params=model.parameters(),
                lr=0.1)
    
    '''
    # To print gradient on backprop
    for param in model.parameters():
      param.register_hook(
          lambda grad: 
            print(grad)
            ) 
    '''
    
    for _ in range(0, num_epochs):
      alpha_optim.zero_grad()
      loss = nn.L1Loss()(model(x), y)
      loss.backward()
      alpha_optim.step()
      print(loss)


if __name__ == '__main__':
  unittest.main()