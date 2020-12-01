# External imports
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import cat

# Internal imports
from alpha import Alpha
from operations import MANDATORY_OPS,  Zero

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
        # Number of filters is specified by channels_start -> thus increasing feature dimension
        nn.Conv2d(channels_in, channels_start, 3, 1, 1, bias=False),
        # Normalization in the regular sense - acts as a regularizer
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
    # Linear transformation without activation function
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
    logits = F.softmax(self.classifer(y), dim=-1)

    return logits

class ModelController(nn.Module):
  '''
  This class is the controller for model and has alpha parameters registered in addition to the weights (weights) parameters automatically registered by Pytorch.

  get_weights -> returns weights parameters

  get_alpha_level(level) -> returns parameter (yes singular, as the whole tensor is wrapped as one parameter) corresponding to alpha_level
  '''
  def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int], primitives: dict, channels_in: int, channels_start: int, stem_multiplier: int,  num_classes: int, loss_criterion):
    '''
    - Initializes member variables
    - Registers alpha parameters by creating a dummy alpha using the constructor and using get_alpha_level to get the alpha for a given level. This tensor is wrapped with nn.Parameter to indicate that is a Parameter for this controller (thus requires gradient computation with respect to itself). This nn.Parameter is added to the nn.ParameterList that is self.alphas.
    - Registers weights parameters by creating a model from aforementioned dummy alpha
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.num_levels = num_levels
    self.num_nodes_at_level = num_nodes_at_level
    self.num_ops_at_level = num_ops_at_level
    self.primitives = primitives
    self.channels_in = channels_in
    self.channels_start = channels_start
    self.stem_multiplier = stem_multiplier
    self.num_classes = num_classes
    self.loss_criterion = loss_criterion

    # Register Alpha parameters
    # Initial Alpha
    alpha = Alpha(
      num_levels=self.num_levels,
      num_nodes_at_level=self.num_nodes_at_level,
      num_ops_at_level=self.num_ops_at_level
    )
    self.alpha = nn.ParameterList() # List of parameters: each alpha_i is a parameter
    for level in range(0, num_levels):
      self.alpha.append(nn.Parameter(alpha.get_alpha_level(level)))
    
    # Initialize model with initial alpha
    self.model = Model(
          alpha=alpha,
          primitives=self.primitives,
          channels_in=self.channels_in,
          channels_start=self.channels_start,
          stem_multiplier=self.stem_multiplier,
          num_classes=self.num_classes)

  def forward(self, x):
    # Initialize alpha from self.alpha parameter list
    alpha = Alpha(
      num_levels=self.num_levels,
      num_nodes_at_level=self.num_nodes_at_level,
      num_ops_at_level=self.num_ops_at_level
    )
    for level in range(0, alpha.num_levels):
      alpha.set_alpha_level(level, self.alpha[level])

    # Initialize model with new alpha
    self.model = Model(
          alpha=alpha,
          primitives=self.primitives,
          channels_in=self.channels_in,
          channels_start=self.channels_start,
          stem_multiplier=self.stem_multiplier,
          num_classes=self.num_classes)

    return self.model(x)

  def loss(self, X, y):
      logits = self.forward(X)
      return self.loss_criterion(logits, y)

  def get_alpha_level(self, level):
    return self.alpha[level]

  def get_weights(self):
    return self.model.parameters()
