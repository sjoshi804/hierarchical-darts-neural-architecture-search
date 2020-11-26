
import unittest
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from alpha import Alpha

'''
FIXME: TODO:
Channels are the big unresolved issue. 

Num channels will change when we have multiple incoming edges to a node since
the way we deal with that is by concatenating the channels - change this? maybe, I don't think so

1)
Could have an extra op at the end of any composite op that brings num channels back to right c_in - but I don't think this is the right way since our network should be able to have many channels in intermediate layers

2)
Maybe let it be as is, figure out a way to pass around num_channels so we don't have an issue where out_channels of one output don't correspond to the in_channels expected when it is treated as input. And at end add an additional module to reduce num features to desired channels_out for whole network

3)
Or 
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

    ''' The ops in operations have everything but channel and stride specified so we want to pass that to them. FIXME: Not yet clear about what this means?
    TODO: Have not yet the type for ops in operations, this must be done to make the above functionality work i.e. we have an operation that is specified in terms of its architecture but just needs channel and stride set. '''
    for op in operations:
      self._ops.append(op(channels, stride, False))
      ## FIXME: The above notation i.e. op(channels, stride, False) doesn't work, change the notation or add support for it

    # Create weights from softmax of alpha_e
    self.weights = F.softmax(alpha_e, dim=-1)

  def forward(self, x):
    return sum(w * op(x) for w, op in zip(self.weights, self._ops))

class HierarchalOperation(nn.module):
  '''
  Takes a list of operations, the number of nodes in the DAG
  alpha here is the alpha for the hierarchal operation as a whole:
  Hence, it is a list of the alpha parameters for each edge in the complete dag of num_nodes

  Analogue of this for pt.darts is https://github.com/khanrc/pt.darts/blob/master/models/search_cells.py
  '''
  def __init__(self, operations, num_nodes, alpha_dag, channels):
    self._ops = nn.ModuleDict()
    self.num_nodes = num_nodes

    # Since alpha dag is in lexicographic order, we need a single index into it
    # This function takes lexicographic ordering
    i = 0
    for node_a in range(0, num_nodes):
      for node_b in range(node_a + 1, num_nodes):
        alpha_e = alpha_dag[i]
        self._ops.add_module(str((node_a, node_b)), MixedOperation(operations, alpha_e, channels, 1)) #TODO: stride always set to 1 for now, perhaps we would like to vary this?
        i += 1

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
        output[edge] = self._ops[str(edge)].forward(input)
    
    # Let the final output be the concatenation of all inputs to the final node
    # TODO: Perhaps we want to add some dropout / reduction here to avoid blowing up the number of features
    output[(self.num_nodes - 1, self.num_nodes)] = cat(tuple([output(prev_node, self.num_nodes - 1) for prev_node in range(0, self.num_nodes - 1)]), dim=0)

    return output[(self.num_nodes - 1, self.num_nodes)]


class Model(nn.module):
  '''
  This class represents the actual resultant neural network.
  
  The constructor requires the architecture parameters and then returns a network that can be used as 
  any other neural network might.
  '''
  def __init__(self, alpha: Alpha, primitives: list, channels_in: int, channels_out: int):
    '''
    Input: alpha - an object of type Alpha
    
    The init function's goal is to create operations of every level.
    Register only the MixedOperation on the the edges of the top most level
    - that will recursively register all the nn.modules and thus get the right omega vector i.e. weights i.e. nn.parameters()
    '''
    # Initialize member variables
    self.alpha = alpha
    self.channels_in = channels_in
    self.channels_out = channels_out

    # Create a dictionary that maps level to list of ops
    self._ops_at_level = {0: primitives}

    # Create all the operations at level num_levels - 1
    # Hence need to consider alpha_0 ... alpha_(num_levels-2)
    for i in range(0, alpha.num_levels - 2):
      ops_i = []
      for op_num in range(0, alpha.num_ops_at_level[i + 1]):
        ops_i.append(HierarchalOperation(self._ops_at_level[i], alpha.num_nodes_at_level[i], alpha.parameters[i][op_num], channels_in))
        # FIXME: channels_in probably doesn't make sense here
      self._ops_at_level[i+1] = ops_i
    
    # Construct top-most level i.e. final architecture
    # Dict from edge tuple to MixedOperation on that edge
    self.top_level_ops = {}
    for node_a in range(0, alpha.num_nodes_at_level[alpha.num_levels - 1]):
      for node_b in range(node_a + 1, alpha.num_nodes_at_level[alpha.num_levels - 1]):
        edge = (node_a, node_b)
        self.top_level_ops[edge] = MixedOperation(self._ops_at_level[alpha.num_levels - 1], alpha.parameters[alpha.num_levels - 1][0][edge], channels_in)
        # FIXME: Channels probably wrong and stride left to default


  def forward(self, x):
    '''
    Take CNN's input and output prediction?

    TODO: Middle part is identical to Hierarchal Operation, 
    but need to ensure the forward takes in input C=channels_in and outputs C=channels_out
    '''
    output = {}
    num_nodes = self.alpha.num_nodes_at_level[self.alpha.num_levels - 1]
    
    for node_a in range(0, ):
      for node_b in range(node_a + 1, num_nodes):
        # For a given edge, determine the input to the starting node
        edge = (node_a, node_b)
        if (node_a == 0): # for node_a = 0, it is trivial, input of entire module
          input = x
        else: # otherwise it is the concatentation of the output of every edge (node, node_a)
          input = []
          for prev_node in range(0, node_a):
            input += output[(prev_node, node_a)]
          input = cat(tuple(input), dim=0) # TODO: Confirm that concatenation along features is what is desired.
        output[edge] = self._ops[str(edge)].forward(input)
    
    # Let the final output be the concatenation of all inputs to the final node
    # TODO: Perhaps we want to add some dropout / reduction here to avoid blowing up the number of features
    output[(num_nodes - 1, num_nodes)] = cat(tuple([output(prev_node, num_nodes - 1) for prev_node in range(0, num_nodes - 1)]), dim=0)

    return output[(num_nodes - 1, num_nodes)]

class TestMixedOperation(unittest.TestCase):
  pass

class TestHierarchalOperation(unittest.TestCase):
  pass

class TestModel(unittest.TestCase):
  pass