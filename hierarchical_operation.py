# External Imports
from numpy import argmax
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
    - ops: dict[stringified tuple for edge -> nn.Module] used to initialize the ModuleDict
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.num_nodes = num_nodes
    self.ops = nn.ModuleDict(ops)

    # Determine channels out - simply a sum of the channels in for the last node
    # We can take this sum by using channels_out property since Mixed operation will have it defined
    self.channels_out = 0
    for prev_node in range(0, num_nodes - 1):
      edge = str((prev_node, num_nodes - 1))
      if edge in self.ops: # Check if edge exists
        self.channels_out += self.ops[edge].channels_out
      else: 
        continue 

  def forward(self, x, x2=None, op_num=0):
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
            edge = str((prev_node, node_a))
            if edge in output: # Ensure edge exists
              input.append(output[edge])
        input = cat(tuple(input), dim=1) 

      for node_b in range(node_a + 1, self.num_nodes):

        edge = str((node_a, node_b))

        # If edge doesn't exist, skip it
        if edge not in self.ops:
          continue
        elif isinstance(self.ops[edge], MixedOperation):
          output[edge] = self.ops[edge].forward(input, op_num=op_num)
        else:
          output[edge] = self.ops[edge].forward(input)
    
    # By extension, final output will be the concatenation of all inputs to the final node
    if type(x2) != type(None): # if top level skip input nodes
      start_node = 2
    else:
      start_node = 0
    return cat(tuple([output[str((prev_node, self.num_nodes - 1))] for prev_node in range(start_node, self.num_nodes - 1)]), dim=1)

  @staticmethod
  def create_dag(level: int, alpha: Alpha, alpha_dags: list, primitives: dict, channels_in_x1: int, channels_in_x2=None, channels=None, is_reduction=False, prev_reduction=False, shared_weights=None, learnt_op=False, input_stride=1):
    '''
    - Recursive funnction to create the computational dag from a given point.
    - Done in this manner to try and ensure that number of channels_in is correct for each operation.
    - Called with top-level dag parameters in the model.__init__ and recursively generates entire model
    TODO: Perhaps add a coin flip that drops paths entirely? Borrwoing from drop_path in darts
    - When using for learnt model extraction ensure that alpha_dags has only one alpha_dag in it
    - When using for weight sharing model training put all alpha_dags that you want shared in this
    '''

        
    # Initialize variables
    num_nodes = alpha.num_nodes_at_level[level]
    dag = {} # from stringified tuple of edge -> nn.Module (to construct nn.ModuleDict from)
    nodes_channels_out = []

    for node_a in range(0, num_nodes-1):
      
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

      ###################
      # Select Operations 
      ###################
      if learnt_op:
        chosen_ops = {}
        # Loop through all node_b >= node_a + offset to create mixed operation on every outgoing edge from node_a 
        for node_b in range(node_a + 1, num_nodes):
          
          # If input node at top level, then do not connect to output node
          # If input node at top level, do not connect to other input node
          if (level == alpha.num_levels - 1) and(node_a < 2) and ((node_b == 1) or (node_b == num_nodes - 1)):
            continue 

          # Determine Operation to Choose
          edge = (node_a, node_b)
          chosen_ops[edge] = int(argmax(alpha_dags[0][edge].cpu().detach()))
        
        ops_to_create = sorted(chosen_ops.values())
      else:
        ops_to_create = range(0, alpha.num_ops_at_level[level])
      base_operations = {}

      if level == 0: 
        # Base case, do not need to recursively create operations at levels below
        primitives.update(MANDATORY_OPS) # Append mandatory ops: identity, zero to primitives
        i = 0
        for key in primitives: 
          if i in ops_to_create: # Avoid creation of unnecessary ops
            base_operations[i] = primitives[key](C=channels_in, stride=stride, affine=False)
          i += 1
      else: 
        # Recursive case, use create_dag to create the list of operations
        if not learnt_op and level == alpha.num_levels - 1:
          base_operations[0] = HierarchicalOperation.create_dag(
            level=level-1,
            alpha=alpha,
            alpha_dags=alpha.parameters[level-1],
            primitives=primitives,
            channels_in_x1=channels_in,
            input_stride=stride,
            learnt_op=learnt_op
          )
        else:
          for op_num in ops_to_create:
            # Skip creation if zero op
            if op_num == alpha.num_ops_at_level[level]:
              continue
            base_operations[op_num] = HierarchicalOperation.create_dag(
              level=level-1,
              alpha=alpha,
              alpha_dags=[alpha.parameters[level-1][op_num]],
              primitives=primitives,
              channels_in_x1=channels_in,
              input_stride=stride,
              learnt_op=learnt_op
            )
          
        # Add zero operation
        base_operations[alpha.num_ops_at_level[level]] = Zero(C_in=channels_in, C_out=base_operations[ops_to_create[0]].channels_out, stride=stride)

      '''
      Determine channels_out
      '''
      # Set channels out (identical for all operations in base_operations)
      nodes_channels_out.append(base_operations[ops_to_create[0]].channels_out)

      '''
      Create mixed operations / Place selected operations on outgoing edges for node_a
      '''
      # Loop through all node_b >= node_a + offset to create mixed operation on every outgoing edge from node_a 
      for node_b in range(node_a + 1, num_nodes):
        
        # If input node at top level, then do not connect to output node
        # If input node at top level, do not connect to other input node
        if (level == alpha.num_levels - 1) and(node_a < 2) and ((node_b == 1) or (node_b == num_nodes - 1)):
          continue 

        # Create mixed operation / Select Learnt Operation on outgiong edge
        edge = (node_a, node_b)  
        if not learnt_op:      
          dag[str(edge)] = MixedOperation(base_operations, [alpha_dag[edge] for alpha_dag in alpha_dags]) 
          ''' Initialize base operation with shared weights if possible '''
          if shared_weights is not None:
            for op_num in base_operations.keys():
              base_operations[op_num].load_state_dict(shared_weights[str(edge)])
        else:
          dag[str(edge)] = base_operations[chosen_ops[edge]]

    '''
    Return HierarchicalOperation created from dag
    '''
    return HierarchicalOperation(alpha.num_nodes_at_level[level], dag)

  # Gets state dictionary for top - 1 level ops
  def get_shared_weights(self):
    shared_weights = {}
    for node_a in range(self.num_nodes):
      for node_b in range(node_a + 1, self.num_nodes):
        edge = str((node_a, node_b))
        if edge in self.ops:
          shared_weights[edge] = self.ops[edge].get_shared_weights()
    return shared_weights
