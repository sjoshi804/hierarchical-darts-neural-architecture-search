# External imports
import unittest
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

# Internal imports
from hdarts import Alpha

'''
Channels / Features are managed in the following way: 
- Output channels = sum of channels on each input edge
- Recursive dag creation to propagate in & out channels correctly in HierarchicalOperation.create_dag()
- The post processing layers condense this by first global pooling to reduce every feature map to a single element and then having a linear mapping to the number of classes (for the classification task)
'''

class MixedOperation(nn.Module):
  '''
  Gets the mixed operation by mixing the operation in accordance with architecture parameters
  TODO: Stride=1 if not specified rn, might want to force specification or change default
  '''
  def __init__(self, operations, alpha_e, channels, stride=1):
    super(MixedOperation, self).__init__()

    '''Create a module list - this registers all the nn.Modules that are children
    of this module - useful as now .parameters i.e. the weights for this class 
    includes the parameters for all of its children recursively'''
    self._ops = nn.ModuleList()

    ''' 
    The ops in operations have everything but channel and stride specified so we want to pass that to them. FIXME: Not yet clear about what this means?
    TODO: Have not yet the type for ops in operations, this must be done to make the above functionality work i.e. we have an operation that is specified in terms of its architecture but just needs channel and stride set. 
    '''
    for op in operations:
      self._ops.append(op(channels, stride, False))
      ## FIXME: The above notation i.e. op(channels, stride, False) doesn't work, change the notation or add support for it

    # Create weights from softmax of alpha_e
    self.weights = F.softmax(alpha_e, dim=-1)

  def forward(self, x):
    return sum(w * op(x) for w, op in zip(self.weights, self._ops))

class HierarchicalOperation(nn.module):
  '''
  Returns a hierarchial operation from a computational dag specified by number of nodes and dict of ops: stringified tuple representing the edge -> nn.Module representing operation on that edge.

  The computational dag for this is created by create_dag inside

  Analogue of this for pt.darts is https://github.com/khanrc/pt.darts/blob/master/models/search_cells.py
  '''
  def __init__(self, num_nodes, ops):
    '''
    Static function create_dag will be called from the model class to initialize the top level 
    '''
    self.num_nodes = num_nodes
    self.ops = ops

  def forward(self, x):
    '''
    Iteratively compute using each edge of the dag
    '''
    output = {}
    for node_a in range(0, self.num_nodes):
      for node_b in range(node_a + 1, self.num_nodes):
        # For a given edge, determine the input to the starting node
        edge = (node_a, node_b)
        if (node_a == 0): # for node_a = 0, it is trivial, input of entire module
          input = x
        else: # otherwise it is the concatentation of the output of every edge (node, node_a)
          input = []
          for prev_node in range(0, node_a):
            input += output[(prev_node, node_a)]
          input = cat(tuple(input), dim=0) # TODO: Confirm that concatenation along features is what is desired.
        output[edge] = self.ops[str(edge)].forward(input)
    
    # Let the final output be the concatenation of all inputs to the final node
    # TODO: Perhaps we want to add some dropout / reduction here to avoid blowing up the number of features
    output[(self.num_nodes - 1, self.num_nodes)] = cat(tuple([output(prev_node, self.num_nodes - 1) for prev_node in range(0, self.num_nodes - 1)]), dim=0)

    return output[(self.num_nodes - 1, self.num_nodes)]

  # Recursive funnction to create the computational dag - needed to do this way to set number of channels correctly
  @staticmethod
  def create_dag(level: int, is_top_level: bool, alpha: Alpha, alpha_dag: dict, primitives: dict, channels_in: int):

    # Initialize variables
    num_nodes = alpha.num_nodes_at_level[level]
    dag = {} # from stringified tuple of edge -> nn.module (to construct nn.ModuleDict from)
    nodes_channels_out = []

    for node_a in range(0, num_nodes):

      # If node not the first then channels in must be computed by sum of channels of input edges
      if (node_a != 0):
        channels_in = sum(nodes_channels_out[:node_a]) #sum of channels out of all nodes < node_a

      for node_b in range(node_a + 1, num_nodes):
        
        edge = (node_a, node_b)

        base_operations = []
        if level == 0: 
          # Base case, do not need to recursively create operations at levels below
          for key in primitives:
            base_operations.append(primitives[key](C=channels_in, stride=1, affine=False))
        else: 
          # Recursive case, use create_dag to create the list of operations
          for op_num in range(0, alpha.num_ops_at_level[level]):
            base_operations.append(HierarchicalOperation.create_dag(
              level=level-1,
              is_top_level=False,
              alpha=alpha,
              alpha_dag=alpha[level-1][op_num],
              primitives=primitives,
              channels_in=channels_in
            ))
        nodes_channels_out[node_a] = base_operations[0].channels_out

        dag[str(edge)] = MixedOperation(base_operations, alpha_dag[edge], channels=channels_in) #TODO: stride left to default=1, change?

    # If top level, then return dag otherwise return Hierarchical Operation to use in higher-level dag
    if (is_top_level):
      return dag
    else:
      return HierarchicalOperation(dag)

class Model(nn.module):
  '''
  This class represents the actual resultant neural network.
  
  The constructor requires the architecture parameters and then returns a network that can be used as 
  any other neural network might.
  '''
  def __init__(self, alpha: Alpha, primitives: list, channels_in: int, channels_start: int, num_classes: int, num_layers: int, stem_multiplier: int, multiplier: int):
    '''
    Input: alpha - an object of type Alpha
    
    Goals: 
    - preprocessing / stem layer(s)
    - postprocessing layer(s)
    - creating operations to place on edges of top-level dag

    '''
    # Initialize member variables
    self.alpha = alpha

    '''
    Preprocessing Neural Network Layers
    '''
    # Create a 'stem' operation that is a sort of preprocessing layer before our hierarchical network
    channels_start = channels_start * stem_multiplier
    self.stem = nn.Sequential(
        nn.Conv2d(channels_in, channels_start, 3, 1, 1, bias=False),
        nn.BatchNorm2d(channels_start)
    )

    '''
    Main Network: Top-Level DAG goes here
    '''

    # Dict from edge tuple to MixedOperation on that edge
    self.top_level_ops = nn.ModuleDict(
      HierarchicalOperation.create_dag
      (
        level=alpha.num_levels - 1,
        is_top_level=True,
        alpha=alpha,
        alpha_dag=alpha.parameters[alpha.num_levels - 1][0],
        primitives=primitives,
        channels_in=channels_start        
      )
    )

    '''
    Postprocessing Neural Network Layers
    '''

    # Penultimate Layer: Global pooling to downsample feature maps to single value
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # Final Layer: Linear classifer to get final prediction, expected output vector of length num_classes
    self.classifier = nn.Linear(channels_start, num_classes)

  def forward(self, x):
    '''
    TODO: Middle part is identical to Hierarchical Operation, 
    but need to ensure the forward takes in input C=channels_in and outputs C=channels_out
    '''
    output = {}
    num_nodes = self.alpha.num_nodes_at_level[self.alpha.num_levels - 1]
    
    for node_a in range(0, ):
      for node_b in range(node_a + 1, num_nodes):
        # For a given edge, determine the input to the starting node
        edge = str((node_a, node_b))
        if (node_a == 0): # for node_a = 0, it is trivial, input of entire module
          input = x
        else: # otherwise it is the concatentation of the output of every edge (node, node_a)
          input = []
          for prev_node in range(0, node_a):
            input += output[(prev_node, node_a)]
          input = cat(tuple(input), dim=0) # TODO: Confirm that concatenation along features is what is desired.
        output[edge] = self.top_level_ops[str(edge)].forward(input)
    
    # Let the final output be the concatenation of all inputs to the final node
    # TODO: Perhaps we want to add some dropout / reduction here to avoid blowing up the number of features
    output[(num_nodes - 1, num_nodes)] = cat(tuple([output(prev_node, num_nodes - 1) for prev_node in range(0, num_nodes - 1)]), dim=0)

    return output[(num_nodes - 1, num_nodes)]

class TestMixedOperation(unittest.TestCase):
  pass

class TestHierarchicalOperation(unittest.TestCase):
  pass

class TestModel(unittest.TestCase):
  pass