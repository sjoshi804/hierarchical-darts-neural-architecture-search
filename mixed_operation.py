import torch.nn as nn
import torch.nn.functional as F

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

    # Initialize weights using alpha_e
    self.alpha_e = alpha_e

    # Module List
    self.ops = nn.ModuleList(operations)

    # Channels out = channels out of any operation in self._ops as all are same, 
    # recursively will have channels_out defined or primitive will have channels_out defined
    self.channels_out = self.ops[0].channels_out

  def forward(self, x):
    '''
    Linear combination of operations scaled by self.weights i.e softmax of the architecture parameters
    '''
    return sum(w * op(x) for w, op in zip(F.softmax(self.alpha_e, dim=-1), self.ops))
