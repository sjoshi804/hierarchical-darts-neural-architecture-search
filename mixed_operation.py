from torch.nn.modules.batchnorm import BatchNorm2d
from operations import Zero
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
    self.alpha_e = nn.ParameterList(alpha_e)

    # Module List
    self.ops = nn.ModuleList([operations[key] for key in sorted(operations.keys())])
    for i in range(len(self.ops)):
      if "pool" in str(type(self.ops[i])).lower():
        self.ops[i] = nn.Sequential(
          self.ops[i],
          BatchNorm2d(self.ops[i].channels_out, affine=False)
        )

  def forward(self, x, op_num=0):
    '''
    Linear combination of operations scaled by self.weights i.e softmax of the architecture parameters
    '''
    if len(self.ops) == 1:
      return sum([w * self.ops[0].forward(x, op_num=op_num) for op_num, w in enumerate(F.softmax(self.alpha_e[0], dim=-1))])
    else:
      return sum(w * op(x) for w, op in zip(F.softmax(self.alpha_e[op_num], dim=-1), self.ops))

  def get_shared_weights(self):
    if len(self.ops) > 2:
      raise Exception("Get shared weights called on Mixed Op with more than 1 op (not including Zero)")
    else:
      return self.ops[0].state_dict()
