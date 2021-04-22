# External imports
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Internal imports
from alpha import Alpha
from hierarchical_operation import HierarchicalOperation

class Model(nn.Module):
  '''
  This class represents the actual resultant neural network.
  
  The constructor requires the architecture parameters and then returns a network that can be used as 
  any other neural network might.
  '''

  def __init__(self, alpha_normal: Alpha, alpha_reduce: Alpha, primitives: dict, channels_in: int, channels_start: int, stem_multiplier: int,  num_classes: int, num_cells: int, shared_weights=None, writer=None, test_mode=False):
    '''
    Input: 
    - alpha - an object of type Alpha
    - primitives - dict[any -> lambda function with inputs C, stride, affine that returns a primitive operation]
    - channels_in - the input channels from the dataset
    - channels_start - the number of channels to start with
    - stem_multiplier - TODO: understand why isn't channels_start * stem_multiplier passed in directly in DARTS implementations
    - num_classes - number of classes that input can be classified into - needed to set up final layers tha map output of hierarchical model to desired output form
    - num_cells - number of cells in model
    - shared_weights - list of state dicts for all the top-1 level ops that are being trained using shared weights
    
    Goals: 
    - preprocessing / stem layer(s)
    - postprocessing layer(s)
    - creating operations to place on edges of top-level dag
    '''
    # Superclass constructor
    super().__init__()

    # Initialize member variables
    self.alpha_normal = alpha_normal
    self.alpha_reduce = alpha_reduce
    self.writer = writer
    self.test_mode = test_mode
    '''
    Pre-processing / Stem Layers
    '''
    if not test_mode:
      # Create a pre-processing / 'stem' operation that is a sort of preprocessing layer before our hierarchical network
      channels_pre_processing = channels_start * stem_multiplier
      self.pre_processing = nn.Sequential(
          # Number of filters is specified by channels_start -> thus increasing feature dimension
          nn.Conv2d(channels_in, channels_pre_processing, 3, 1, 1, bias=False),
          # Normalization in the regular sense - acts as a regularizer
          nn.BatchNorm2d(channels_pre_processing)
      )

    '''
    Main Network: Top-Level DAGs for Cells created here
    '''
    # List of modules that holds all the cells
    self.main_net = nn.ModuleList()

    # At 1/3 and 2/3 num_cells, reduction cells are inserted
    reduction_cell_indices = [num_cells//3, (num_cells//3)*2]

    # Initialize channels
    curr_channels = channels_start 

    # Create cells
    for i in range(0, num_cells):
      # Determine channels
      if (i - 2) < 0:
        prev_prev_channels = channels_pre_processing
      else:
        prev_prev_channels = self.main_net[i - 2].channels_out
      if (i - 1) < 0:
        prev_channels = channels_pre_processing
      else:
        prev_channels = self.main_net[i - 1].channels_out
      
      if i in reduction_cell_indices:
        # Reduction Cell - halve feature map, double # features
        curr_channels *= 2
        self.main_net.append(HierarchicalOperation.create_dag(
          level=alpha_reduce.num_levels - 1,
          alpha=alpha_reduce,
          alpha_dag=alpha_reduce.parameters[alpha_reduce.num_levels - 1][0],
          primitives=primitives,
          channels_in_x1=prev_prev_channels,
          channels_in_x2=prev_channels,
          channels=curr_channels,
          is_reduction=True,
          prev_reduction=(i-1 in reduction_cell_indices), 
          shared_weights=None if shared_weights is None else shared_weights[i] 
        ))
      else:
        # Normal Cell
        self.main_net.append(HierarchicalOperation.create_dag(
          level=alpha_normal.num_levels - 1,
          alpha=alpha_normal,
          alpha_dag=alpha_normal.parameters[alpha_normal.num_levels - 1][0],
          primitives=primitives,
          channels_in_x1=prev_prev_channels,
          channels_in_x2=prev_channels,
          channels=curr_channels,
          is_reduction=False,
          prev_reduction=(i-1 in reduction_cell_indices),
          shared_weights=None if shared_weights is None else shared_weights[i] 
        ))
      print("Channels In / Out for Cells")
      print("Cell", i, "C_in", curr_channels, "C_out", self.main_net[i].channels_out)

    '''
    Post-processing Layers
    '''
    if not test_mode:
      # Penultimate Layer: Global average pooling to downsample feature maps to single value
      self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

      # Final Layer: Linear classifer to get final prediction, expected output vector of length num_classes
      # Linear transformation without activation function
      self.classifer = nn.Linear(self.main_net[-1].channels_out, num_classes) 

  def forward(self, x): 
    '''
    This function applies the pre-processing layers to the input first, then the actual model using the top-level dag, then applies the post-processing layers that first by using global_avg_pooling downsample feature maps to single values, then this is flattened and finally a linear classifer uses this to output a prediction.
    '''

    '''
    Pre-processing / Stem Layers
    '''
    if not self.test_mode:
      if (torch.cuda.is_available()):
        x = x.cuda()
      x = self.pre_processing(x)

    '''
    Main model 
    '''
    output = []
    for i in range(0, len(self.main_net)):
      # Use stem for input if no previous cells
      if (i - 2 < 0):
        x_prev_prev = x
      else:
        x_prev_prev = output[i - 2]
      if (i - 1 < 0):
        x_prev = x 
      else:
        x_prev = output[i - 1]
      # Append to output
      output.append(self.main_net[i].forward(x_prev_prev, x_prev))
    y = output[-1]

    '''
    Post-processing Neural Network Layers
    ''' 
    if not self.test_mode:
      # Global Avg Pooling
      y = self.global_avg_pooling(y)

      # Flatten
      y = y.view(y.size(0), -1) 

      # Classifier
      logits = self.classifer(y)

      return logits
    else:
      return y

  def get_shared_weights(self):
    shared_weights = [] # list of dictionaries that map edges to state_dicts
    for cell in self.main_net:
      shared_weights.append(cell.get_shared_weights())
    return shared_weights
