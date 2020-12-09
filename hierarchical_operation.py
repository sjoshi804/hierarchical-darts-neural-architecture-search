# External Imports
from torch import cat  
import torch.nn as nn 

# Internal Imports
from alpha import Alpha
from mixed_operation import MixedOperation
from operations import MANDATORY_OPS, Zero

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