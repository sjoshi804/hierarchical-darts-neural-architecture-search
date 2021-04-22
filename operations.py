from torch import tensor, cat
import torch.nn as nn

'''
Input Dim: (N, C_in, H_in, W_in)
Output Dim: (N, C_out, H_out, W_out)

N is the number of samples in a batch
C is the number of channels - in the case of color images this is RGB
H_in, W_in - dimensions of the 2d matrix for a given channel (also called feature)
H_out, W_out are determined by stride and H_in, W_in

Stride determines at what stride we will look at H_in, W_in

Affine: Matrix multiplication of input and weights - Ask Professor? - can ignore

Useful Links:
- https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- https://datascience.stackexchange.com/questions/13405/what-is-affine-transformation-in-regards-to-neural-networks

'''
SIMPLE_OPS = {
  "double": lambda C, stride, affine: Double(C, stride),
  "triple": lambda C, stride, affine: Triple(C, stride)
}

# FIXME: Fix this to make more sense
VAE_OPS = {
  "conv_3x3_4x_bn_relu": lambda C, stride, affine: ConvBNReLu(C, 4*C, 3, 1, 1)
}
LEN_VAE_OPS = len(VAE_OPS)

LEN_SIMPLE_OPS = len(SIMPLE_OPS)

MANDATORY_OPS = {
  "identity": lambda C, stride, affine: Identity(C, stride),
  "zero": lambda C, stride, affine: Zero(C, C, stride)
}

# TODO: Try replacing the larger convolutions with stacking of the smaller ones instead
OPS = {
  'avg_pool_3x3' : lambda C, stride, affine: AvgPool2d(C, C, 3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: MaxPool2d(C, C, 3, stride=stride, padding=1, count_include_pad=False), #(3, stride=stride, padding=1), #add batch normalization here
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  #'2_stacked_sep_conv_3x3': lambda C, stride, affine: StackedSepConv(C, C, 3, stride, 1, 2),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  #'3_stacked_sep_conv_3x3': lambda C, stride, affine: StackedSepConv(C, C, 3, stride, 1, 3),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  #'2_stacked_dil_conv_3x3': lambda C, stride, affine: StackedDilConv(C, C, 3, stride, 2, 2, 2),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: Conv7x1_1x7(C, stride, affine=affine)
}

LEN_OPS = len(OPS)

class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.channels_in = C_in
    self.channels_out = C_out

  def forward(self, x, op_num=None):
    if (self.channels_in < self.channels_out):
      feature_map = [[0.] * (len(x[0][0][0])//self.stride)] * (len(x[0][0])//self.stride)
      output = tensor(([[feature_map] * (self.channels_out)] * len(x)))
      return output.to(x.device)
    elif (self.channels_in > self.channels_out ):
      raise Exception("Assumption violated: channels_in > channels_out")

    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
      

class Identity(nn.Module):

  def __init__(self, C, stride):
    super(Identity, self).__init__()
    self.stride = stride
    self.channels_out = C

  def forward(self, x):
    if self.stride == 1:
      return x
    return x[:,:,::self.stride,::self.stride]

class Double(nn.Module):

  def __init__(self, C, stride):
    super(Double, self).__init__()
    self.stride = stride
    self.channels_out = C

  def forward(self, x):
    if self.stride == 1:
      return x.mul(2.)
    return x[:,:,::self.stride,::self.stride].mul(2.)

class Triple(nn.Module):

  def __init__(self, C, stride):
    super(Triple, self).__init__()
    self.stride = stride
    self.channels_out = C

  def forward(self, x):
    if self.stride == 1:
      return x.mul(3.)
    return x[:,:,::self.stride,::self.stride].mul(3.)

class Conv7x1_1x7(nn.Module):
  def __init__(self, C, stride, affine):
    super().__init__()
    self.op = nn.Sequential(
                            nn.ReLU(inplace=False),
                            nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
                            nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
                            nn.BatchNorm2d(C, affine=affine)
                            )
    self.channels_out = C  
  
  def forward(self, x):
    return self.op(x)
    

class AvgPool2d(nn.Module): 
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False, count_include_pad=False):
    super(AvgPool2d, self).__init__()
    self.op = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=count_include_pad)
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)

class MaxPool2d(nn.Module): 
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False, count_include_pad=False):
    super(MaxPool2d, self).__init__()
    self.op = nn.MaxPool2d(3, stride=stride, padding=padding) #(3, stride=stride, padding=padding, count_include_pad=count_include_pad)
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)

class ConvBNReLu(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
    super(ConvBNReLu, self).__init__()
    self.op =  nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(C_out),
            nn.ReLU()
        )
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.channels_out = C_out

  def forward(self, x):
    return self.op(x)

class StackedSepConv(nn.Module):
  def init(self, C_in, C_out, kernel_size, stride, padding, num_stacked, affine=False):
      super(StackedSepConv, self).__init__()
      self.ops = nn.ModuleList()
      for i in range(num_stacked):
        if i == 0:
          channels_in = C_in
        else:
          channels_in = self.ops[i-1].channels_out
        self.ops.append(SepConv(channels_in, C_out, kernel_size, stride, padding, affine=affine))

  def forward(self, x):
    for op in self.ops:
      x = op(x)
    return x

class StackedDilConv(nn.Module):
  def init(self, C_in, C_out, kernel_size, stride, padding, dilation, num_stacked, affine=False):
      super(StackedDilConv, self).__init__()
      self.ops = nn.ModuleList()
      for i in range(num_stacked):
        if i == 0:
          channels_in = C_in
        else:
          channels_in = self.ops[i-1].channels_out
        self.ops.append(DilConv(channels_in, C_out, kernel_size, stride, padding, dilation, affine=affine))

  def forward(self, x):
    for op in self.ops:
      x = op(x)
    return x

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=False):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)
