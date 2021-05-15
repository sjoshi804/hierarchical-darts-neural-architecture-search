# External Imports
from util import drop_path
from numpy import argmax
from torch import cat  
import torch.nn as nn 

# Internal Imports
from alpha import Alpha
from mixed_operation import MixedOperation
from operations import FactorizedReduce, Identity, MANDATORY_OPS, StdConv, Zero

# String Constants
PREPROC_X = "preproc_x"
PREPROC_X2 = "preproc_x2"

# DROP PROB
DROP_PROB=0.2

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
  def __init__(self, num_nodes, ops, channels_in, concatenate_output=False, learnt_op=False, darts_sim=False):
    '''
    - num_nodes
    - ops: dict[stringified tuple for edge -> nn.Module] used to initialize the ModuleDict
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.num_nodes = num_nodes
    self.ops = nn.ModuleDict(ops)
    self.concatenate_output = concatenate_output
    self.learnt_op = learnt_op
    self.darts_sim = darts_sim 

    # Channels Out
    self.channels_out = channels_in * (num_nodes - 3) if concatenate_output else channels_in

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

    for node_a in range(0, self.num_nodes - 1):
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
            input.append(output[edge])
        input = sum(input)

      for node_b in range(node_a + 1, self.num_nodes):

        edge = str((node_a, node_b))

        # If edge doesn't exist, skip it
        if (type(x2) != type(None)) and (node_a < 2 and (node_b == 1 or node_b == self.num_nodes - 1)):
          continue
        elif (type(x2) != type(None)) and (node_b == self.num_nodes - 1):
          output[edge] = input
        elif isinstance(self.ops[edge], MixedOperation):
          output[edge] = self.ops[edge].forward(input, op_num=op_num)
        else:
          # If not at top level maybe drop path, else don't #FIXME: FOR DARTS WE WANT DROP PROB AT TOP LEVEL
          if self.learnt_op and (self.darts_sim or type(x2) == type(None)) and not isinstance(self.ops[edge], Identity): 
            output[edge] = drop_path(self.ops[edge].forward(input), DROP_PROB)
          else:
            output[edge] = self.ops[edge].forward(input)
    
    # By extension, final output will be the concatenation of all inputs to the final node
    if type(x2) != type(None): # if top level skip input nodes
      start_node = 2
    else:
      start_node = 0

    # Concatenate Output only if top level op
    if self.concatenate_output:
      return cat(tuple([output[str((prev_node, self.num_nodes - 1))] for prev_node in range(start_node, self.num_nodes - 1)]), dim=1)
    else:
      return sum([output[str((prev_node, self.num_nodes - 1))] for prev_node in range(start_node, self.num_nodes - 1)])


  @staticmethod
  def create_dag(level: int, alpha: Alpha, alpha_dags: list, primitives: dict, channels_in_x1: int, channels_in_x2=None, channels=None, is_reduction=False, prev_reduction=False, shared_weights=None, learnt_op=False, input_stride=1):
    '''
    - Recursive funnction to create the computational dag from a given point.
    - Done in this manner to try and ensure that number of channels_in is correct for each operation.
    - Called with top-level dag parameters in the model.__init__ and recursively generates entire model
    - When using for learnt model extraction ensure that alpha_dags has only one alpha_dag in it
    - When using for weight sharing model training put all alpha_dags that you want shared in this
    '''

        
    # Initialize variables
    num_nodes = alpha.num_nodes_at_level[level]
    dag = {} # from stringified tuple of edge -> nn.Module (to construct nn.ModuleDict from)

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
      Determine Pre-Processing If Necessary
      '''
      if alpha.num_levels - 1 == level:
        if prev_reduction: 
          dag[PREPROC_X] = FactorizedReduce(channels_in_x1, channels, affine=False)
        else:
          dag[PREPROC_X] = StdConv(channels_in_x1, channels, 1, 1, 0, affine=False)
        dag[PREPROC_X2] = StdConv(channels_in_x2, channels, 1, 1, 0, affine=False)

      '''
      Determine Channels In
      '''
      if channels is None:
        channels = channels_in_x1
        
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
          if (level == alpha.num_levels - 1) and ((node_a < 2 and node_b == 1) or (node_b == num_nodes - 1)):
            continue 

          # Determine Operation to Choose
          edge = (node_a, node_b)
          # If primitive level, then last op is zero - do not include
          if level == 0:
            alpha_candidates = alpha_dags[0][edge].cpu().detach()[:-1]
          else:
            alpha_candidates = alpha_dags[0][edge].cpu().detach()
          chosen_ops[edge] = int(argmax(alpha_candidates))
        
        ops_to_create = sorted(set(chosen_ops.values()))

        '''
        Top-K Sparsification if DARTS SIM
        '''
        if alpha.num_levels == 1:
          incoming_max_alpha = {}
          for node_b in range(node_a + 1, num_nodes):
            incoming_edge = (node_a, node_b)
            if incoming_edge in alpha_dags[0]:
              incoming_max_alpha[incoming_edge] = max(alpha_dags[0][incoming_edge].cpu().detach()[:-1])
          edges_to_keep = sorted(incoming_max_alpha, key=incoming_max_alpha.get)[-2:]
      else:
        ops_to_create = range(0, alpha.num_ops_at_level[level])
      base_operations = {}
      
      # Variable to store number of channels out for Zero Op
      if level == 0: 
        # Base case, do not need to recursively create operations at levels below
        primitives.update(MANDATORY_OPS) # Append mandatory ops: identity, zero to primitives
        i = 0
        for key in primitives: 
            base_operations[i] = primitives[key](C=channels, stride=stride, affine=False)
            i += 1
      else: 
        # Recursive case, use create_dag to create the list of operations
        if not learnt_op and level == alpha.num_levels - 1:
          base_operations[0] = HierarchicalOperation.create_dag(
            level=level-1,
            alpha=alpha,
            alpha_dags=alpha.parameters[level-1],
            primitives=primitives,
            channels_in_x1=channels,
            input_stride=stride,
            learnt_op=learnt_op
          )
        else:
          for op_num in ops_to_create:
            # Skip creation if zero op
            base_operations[op_num] = HierarchicalOperation.create_dag(
              level=level-1,
              alpha=alpha,
              alpha_dags=[alpha.parameters[level-1][op_num]],
              primitives=primitives,
              channels_in_x1=channels,
              input_stride=stride,
              learnt_op=learnt_op
            )
      
      '''
      Create mixed operations / Place selected operations on outgoing edges for node_a
      '''
      # Loop through all node_b >= node_a + offset to create mixed operation on every outgoing edge from node_a 
      for node_b in range(node_a + 1, num_nodes):
        
        # If input node at top level, then do not connect to output node
        # If input node at top level, do not connect to other input node
        if (level == alpha.num_levels - 1) and ((node_a < 2 and node_b == 1) or (node_b == num_nodes - 1)):
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
          if alpha.num_levels != 1:
            dag[str(edge)] = base_operations[chosen_ops[edge]]
          else: # DARTS SIM - TOP K SPARSIFICATION  #FIXME: Hacky Fix
            if node_b == num_nodes-1 or edge in edges_to_keep:
              dag[str(edge)] = base_operations[chosen_ops[edge]]
    '''        
    Return HierarchicalOperation created from dag
    '''
    return HierarchicalOperation(alpha.num_nodes_at_level[level], dag, channels, level==alpha.num_levels - 1, learnt_op=learnt_op)

  # Gets state dictionary for top - 1 level ops
  def get_shared_weights(self):
    shared_weights = {}
    for node_a in range(self.num_nodes):
      for node_b in range(node_a + 1, self.num_nodes):
        edge = str((node_a, node_b))
        if edge in self.ops:
          shared_weights[edge] = self.ops[edge].get_shared_weights()
    return shared_weights
