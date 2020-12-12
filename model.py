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

  def __init__(self, alpha: Alpha, primitives: dict, channels_in: int, channels_start: int, stem_multiplier: int,  num_classes: int, writer=None, test_mode=False):
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
    self.writer = writer
    self.test_mode = test_mode
    '''
    Pre-processing / Stem Layers
    '''
    if not test_mode:
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
    if not test_mode:
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
    if not self.test_mode:
      if (torch.cuda.is_available()):
        x = x.cuda()
      x = self.pre_processing(x)

    '''
    Main model - identical to HierarchicalOperation.forward in this section
    '''

    y = self.top_level_op(x)
    if self.writer is not None:
      pass
      #self.writer.add_graph(self.top_level_op, x)

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