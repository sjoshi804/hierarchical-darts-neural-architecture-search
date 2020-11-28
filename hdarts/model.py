# External imports
import unittest
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, tensor, zeros

# Internal imports
from alpha import Alpha
from operations import SIMPLE_OPS, MANDATORY_OPS, LEN_SIMPLE_OPS, Zero

'''
Channels / Features are managed in the following way: 
- Output channels = sum of channels on each input edge
- Recursive dag creation to propagate in & out channels correctly in HierarchicalOperation.create_dag()
- The post processing layers condense this by first global pooling to reduce every feature map to a single element and then having a linear mapping to the number of classes (for the classification task)
'''

class MixedOperation(nn.Module):
  '''
  - Gets the mixed operation by mixing the operation in accordance with architecture parameters
  - Operations already have channels / stride set and just need to be applied now.
  '''

  def __init__(self, operations, alpha_e):
    '''
    - Create a module list: this registers all the nn.Modules that are children
    of this module (the parameters for this class 
    includes the parameters for all of its children recursively)
    - Set self.weights to the softmax of the architecture parameters for the edge
    '''
    # Superclass constructor
    super().__init__()

    # Module List
    self.ops = nn.ModuleList(operations)

    # Create weights from softmax of alpha_e
    self.weights = F.softmax(alpha_e, dim=-1)

    # Channels out = channels out of any operation in self._ops as all are same, 
    # recursively will have channels_out defined or primitive will have channels_out defined
    self.channels_out = self.ops[0].channels_out

  def forward(self, x):
    '''
    Linear combination of operations scaled by self.weights i.e softmax of the architecture parameters
    '''
    return sum(w * op(x) for w, op in zip(self.weights, self.ops))

class HierarchicalOperation(nn.Module):
  '''
  Returns a hierarchical operation from a computational dag specified by number of nodes and dict of ops: stringified tuple representing the edge -> nn.Module representing operation on that edge.

  The computational dag for this is created by create_dag inside

  Analogue of this for pt.darts is https://github.com/khanrc/pt.darts/blob/master/models/search_cells.py
  '''
  def __init__(self, num_nodes, ops):
    '''
    - num_nodes
    - ops: dict[stringified tuple for edge -> nn.module] used to initialize the ModuleDict
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.num_nodes = num_nodes
    self.ops = nn.ModuleDict(ops)

    # Determine channels out - simply a sum of the channels in for the last node
    # We can take this sum by using channels_out property since Mixed operation will have it defined
    self.channels_out = sum(self.ops[str((prev_node, num_nodes - 1))].channels_out for prev_node in range(0, num_nodes - 1))

  def forward(self, x):
    '''
    Iteratively compute using each edge of the dag
    '''
    output = {}

    for node_a in range(0, self.num_nodes):
      for node_b in range(node_a + 1, self.num_nodes):

        # For a given edge, determine the input to the starting node
        edge = (node_a, node_b)

        if (node_a == 0): 
          # for node_a = 0, it is trivial, input of entire module
          input = x

        else: 
          # otherwise it is the concatentation of the output of every edge (node, node_a)
          input = []

          for prev_node in range(0, node_a):
            input.append(output[(prev_node, node_a)])

          input = cat(tuple(input), dim=1) 
          # TODO: Confirm that concatenation along features is what is desired.

        output[edge] = self.ops[str(edge)](input)
    
    # By extension, final output will be the concatenation of all inputs to the final node
    # TODO: Perhaps we want to add some dropout / reduction here to avoid blowing up the number of features
    return cat(tuple([output[(prev_node, self.num_nodes - 1)] for prev_node in range(0, self.num_nodes - 1)]), dim=1)

  @staticmethod
  def create_dag(level: int, alpha: Alpha, alpha_dag: dict, primitives: dict, channels_in: int):
    '''
    - Recursive funnction to create the computational dag from a given point.
    - Done in this manner to try and ensure that number of channels_in is correct for each operation.
    - Called with top-level dag parameters in the model.__init__ and recursively generates entire model
    TODO: Perhaps add a coin flip that drops paths entirely? Borrwoing from drop_path in darts
    '''

    # Initialize variables
    num_nodes = alpha.num_nodes_at_level[level]
    dag = {} # from stringified tuple of edge -> nn.module (to construct nn.ModuleDict from)
    nodes_channels_out = []

    for node_a in range(0, num_nodes):

      '''
      Determine channels_in
      '''
      # If not the first node then channels_in must be computed by sum of channels of input edges
      if (node_a != 0):
        channels_in = sum(nodes_channels_out[:node_a]) 

      '''
      Determine base set of operations
      '''

      base_operations = []

      if level == 0: 
        # Base case, do not need to recursively create operations at levels below
        primitives.update(MANDATORY_OPS) # Append mandatory ops: identity, zero to primitives
        for key in primitives: 
          base_operations.append(primitives[key](C=channels_in, stride=1, affine=False))
          #TODO: stride left to default=1, change?

      else: 
        # Recursive case, use create_dag to create the list of operations
        for op_num in range(0, alpha.num_ops_at_level[level]):
          base_operations.append(HierarchicalOperation.create_dag(
            level=level-1,
            alpha=alpha,
            alpha_dag=alpha.parameters[level-1][op_num],
            primitives=primitives,
            channels_in=channels_in
          ))

        # Append zero operation
        base_operations.append(Zero(C_in=channels_in, C_out=base_operations[0].channels_out, stride=1))

      '''
      Determine channels_out
      '''
      # Set channels out (identical for all operations in base_operations)
      nodes_channels_out.append(base_operations[0].channels_out)

      '''
      Create mixed operations on outgoing edges for node_a
      '''
      # Loop through all node_b > node_a to create mixed operation on every outgoing edge from node_a 
      for node_b in range(node_a + 1, num_nodes):
        
        # Create mixed operation on outgiong edge
        edge = (node_a, node_b)
        dag[str(edge)] = MixedOperation(base_operations, alpha_dag[edge]) 

    '''
    Return HierarchicalOperation created from dag'
    '''
    return HierarchicalOperation(alpha.num_nodes_at_level[level], dag)

class Model(nn.Module):
  '''
  This class represents the actual resultant neural network.
  
  The constructor requires the architecture parameters and then returns a network that can be used as 
  any other neural network might.
  '''

  def __init__(self, alpha: Alpha, primitives: dict, channels_in: int, channels_start: int, stem_multiplier: int,  num_classes: int):
    '''
    Input: 
    - alpha - an object of type Alpha
    - primitives - dict[any -> lambda function with inputs C, stride, affine that returns a primitive operation]
    - channels_in - the input channels from the dataset
    - channels_start - the number of channels to start with
    - stem_multiplier - TODO: understand why isn't channels_start * stem_multiplier passed in directly in DARTS implementations
    - num_classes - number of classes that input can be classified into - needed to set up final layers tha map output of hierarchical model to desired output form
    
    Goals: 
    - preprocessing / stem layer(s)
    - postprocessing layer(s)
    - creating operations to place on edges of top-level dag
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.alpha = alpha

    '''
    Pre-processing / Stem Layers
    '''
    # Create a pre-processing / 'stem' operation that is a sort of preprocessing layer before our hierarchical network
    channels_start = channels_start * stem_multiplier
    self.pre_processing = nn.Sequential(
        nn.Conv2d(channels_in, channels_start, 3, 1, 1, bias=False),
        nn.BatchNorm2d(channels_start)
    )

    '''
    Main Network: Top-Level DAG created here
    '''

    # Dict from edge tuple to MixedOperation on that edge
    self.top_level_op = HierarchicalOperation.create_dag(
      level=alpha.num_levels - 1,
      alpha=alpha,
      alpha_dag=alpha.parameters[alpha.num_levels - 1][0],
      primitives=primitives,
      channels_in=channels_start        
    )

    '''
    Post-processing Layers
    '''
    # Penultimate Layer: Global average pooling to downsample feature maps to single value
    self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    # Final Layer: Linear classifer to get final prediction, expected output vector of length num_classes
    self.classifer = nn.Linear(self.top_level_op.channels_out, num_classes) 

  def forward(self, x): 
    '''
    This function applies the pre-processing layers to the input first, then the actual model using the top-level dag, then applies the post-processing layers that first by using global_avg_pooling downsample feature maps to single values, then this is flattened and finally a linear classifer uses this to output a prediction.
    '''

    '''
    Pre-processing / Stem Layers
    '''
    x = self.pre_processing(x)

    '''
    Main model - identical to HierarchicalOperation.forward in this section
    '''

    y = self.top_level_op(x)

    '''
    Post-processing Neural Network Layers
    ''' 

    # Global Avg Pooling
    y = self.global_avg_pooling(y)

    # Flatten
    y = y.view(y.size(0), -1) 

    # Classifier
    logits = self.classifer(y)

    return logits

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


class TestModel(unittest.TestCase):
  pass

if __name__ == '__main__':
  unittest.main()