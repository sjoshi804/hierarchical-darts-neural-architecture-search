from torch import cat, tensor
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

LEN_SIMPLE_OPS = 2

MANDATORY_OPS = {
  "identity": lambda C, stride, affine: Identity(C, stride),
  "zero": lambda C, stride, affine: Zero(C, C, stride)
}


# TODO: Check if this is correct.
class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.channels_in = C_in
    self.channels_out = C_out

  def forward(self, x):
    if (self.channels_in < self.channels_out):
      #Add extra channels to make channels_out sufficient
      dummy_input = tensor([ [[[0]]] for i in range(0, len(x))])
      x = cat(tuple([x] + [ dummy_input for i in range(0, self.channels_out - self.channels_in) ]), dim=-1)
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

'''
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), #add batch normalization here
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
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

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

'''