# External Imports
from torch import cat  
import torch.nn as nn 

# Internal Imports
from alpha import Alpha
from mixed_operation import MixedOperation
from operations import FactorizedReduce, MANDATORY_OPS, StdConv, Zero

# String Constants
PREPROC_X = "preproc_x"
PREPROC_X2 = "preproc_x2"

'''
Channels / Features are managed in the following way: 
- Output channels = sum of channels on each input edge
- Recursive dag creation to propagate in & out channels correctly in HierarchicalOperation.create_dag()
- The post processing layers condense this by first global pooling to reduce every feature map to a single element and then having a linear mapping to the number of classes (for the classification task)
'''

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

  def forward(self, x, x2=None):
    '''
    Iteratively compute using each edge of the dag
    '''
    output = {}

    # Apply preprocessing if applicable
    if PREPROC_X in self.ops:
      x = self.ops[PREPROC_X].forward(x)
    if PREPROC_X2 in self.ops:
      x2 = self.ops[PREPROC_X2].forward(x2)

    for node_a in range(0, self.num_nodes):
      # For a given edge, determine the input to the starting node
      if (node_a == 0): 
        # for node_a = 0, it is trivial, input of entire module / first input
        input = x
      elif (node_a == 1 and type(x2) != type(None)):
        # if top level, then x2 provided then use for second node
        input = x2
      else: 
        # otherwise it is the concatentation of the output of every edge (node, node_a)
        input = []
        for prev_node in range(0, node_a):
          input.append(output[(prev_node, node_a)])
        input = cat(tuple(input), dim=1) 

      for node_b in range(node_a + 1, self.num_nodes):

        edge = (node_a, node_b)

        # If edge doesn't exist, skip it
        if str(edge) not in self.ops:
          continue
        else:       
          output[edge] = self.ops[str(edge)].forward(input)
    
    # By extension, final output will be the concatenation of all inputs to the final node
    return cat(tuple([output[(prev_node, self.num_nodes - 1)] for prev_node in range(0, self.num_nodes - 1)]), dim=1)

  @staticmethod
  def create_dag(level: int, alpha: Alpha, alpha_dag: dict, primitives: dict, channels_in_x1: int, channels_in_x2=None, channels=None, is_reduction=False, prev_reduction=False, input_stride=1):
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
      Determine stride
      '''
      if (level == alpha.num_levels - 1 and is_reduction and node_a < 2):
        stride = 2
      elif (node_a == 0):
        stride = input_stride
      else: 
        stride = 1

      '''
      Determine channels_in
      '''
      if node_a == 0:
        channels_in = channels_in_x1 
      else:
        channels_in = sum(nodes_channels_out[:node_a]) 

      '''
      Determine Pre-Processing If Necessary
      '''
      if alpha.num_levels - 1 == level:
        if prev_reduction: 
          dag[PREPROC_X] = FactorizedReduce(channels_in_x1, channels, affine=False)
        else:
          dag[PREPROC_X] = StdConv(channels_in_x1, channels, 1, 1, 0, affine=False)
        dag[PREPROC_X2] = StdConv(channels_in_x2, channels, 1, 1, 0, affine=False)
        # If input node
        if node_a < 2:
          channels_in = channels

      '''
      Determine base set of operations
      '''
      base_operations = []

      if level == 0: 
        # Base case, do not need to recursively create operations at levels below
        primitives.update(MANDATORY_OPS) # Append mandatory ops: identity, zero to primitives
        for key in primitives: 
          base_operations.append(primitives[key](C=channels_in, stride=stride, affine=False))

      else: 
        # Recursive case, use create_dag to create the list of operations
        for op_num in range(0, alpha.num_ops_at_level[level]):
          base_operations.append(HierarchicalOperation.create_dag(
            level=level-1,
            alpha=alpha,
            alpha_dag=alpha.parameters[level-1][op_num],
            primitives=primitives,
            channels_in_x1=channels_in,
            input_stride=stride
          ))

        # Append zero operation
        base_operations.append(Zero(C_in=channels_in, C_out=base_operations[0].channels_out, stride=stride))

      '''
      Determine channels_out
      '''
      # Set channels out (identical for all operations in base_operations)
      nodes_channels_out.append(base_operations[0].channels_out)

      '''
      Create mixed operations on outgoing edges for node_a
      '''
      # If top level and first node, then we mustn't connect node 0 to node 1 as they are both input nodes
      if (node_a == 0 and level == alpha.num_levels-1):
        offset = 2
      else:
        offset = 1

      # Loop through all node_b >= node_a + offset to create mixed operation on every outgoing edge from node_a 
      for node_b in range(node_a + offset, num_nodes):
        
        # If input node at top level, then do not connect to output node
        if (node_a < 2) and node_b == num_nodes - 1:
          continue 

        # Create mixed operation on outgiong edge
        edge = (node_a, node_b)        
        dag[str(edge)] = MixedOperation(base_operations, alpha_dag[edge]) 

    '''
    Return HierarchicalOperation created from dag'
    '''
    return HierarchicalOperation(alpha.num_nodes_at_level[level], dag)
