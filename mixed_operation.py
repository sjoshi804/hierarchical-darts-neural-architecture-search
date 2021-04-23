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

    # Channels out = channels out of any operation in self._ops as all are same, 
    # recursively will have channels_out defined or primitive will have channels_out defined
    self.channels_out = self.ops[0].channels_out

  def forward(self, x, op_num=0):
    '''
    Linear combination of operations scaled by self.weights i.e softmax of the architecture parameters
    '''
    if len(self.ops) == 2:
      # Get operation that is not zero
      for _,value in self.ops:
        if not isinstance(value, Zero):
          op = value 
        else:
          zero_op = value

      softmaxed_weights = F.softmax(self.alpha_e[op_num], dim=-1)

      # Need to handle zero operation specially here
      output = [w * op.forward(x, op_num) for w, op_num in zip(softmaxed_weights[:-1], range(len(self.ops) - 1))]
      output.append(softmaxed_weights[-1] * zero_op)
      return sum(output)
    else:
      return sum(w * op(x) for w, op in zip(F.softmax(self.alpha_e[op_num], dim=-1), self.ops))

  def get_shared_weights(self):
    if len(self.ops) > 2:
      raise Exception("Get shared weights called on Mixed Op with more than 1 op (not including Zero)")
    else:
      return self.ops[0].state_dict()