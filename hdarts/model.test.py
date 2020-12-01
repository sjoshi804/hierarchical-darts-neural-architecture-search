import unittest
from torch import cat, equal, tensor, zeros

#Internal imports
from alpha import Alpha
from model import MixedOperation, HierarchicalOperation
from operations import SIMPLE_OPS, LEN_SIMPLE_OPS

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

  #Leon rewrite this line by line
  #LEON, The operations are equally weighted
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

    #0(Zero) + 1(Identity) + 2(Double) + 3(Triple) = 6
    #6 / 4 types of operations = 1.5


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

  #Leon WRITE THIS.  If you create another file to create the tests 
  # have it print out line by line
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

    # TODO: Put the correct y here
    y = tensor([
      # feature 1
      [[
        [1.5, 1.5],
        [1.5, 1.5]
      ]],
      # feature 2
      [[
        [2.25, 2.25],
        [2.25, 2.25]
      ]]
    ])

    print(hierarchical_op(x))

    assert(y.equal(hierarchical_op(x)))


class TestModel(unittest.TestCase):
  pass

if __name__ == '__main__':
  unittest.main()