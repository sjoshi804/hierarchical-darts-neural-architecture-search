from functools import partial
import argparse
import os
import os 
import torch

# Internal Imports
from operations import LEN_OPS


# TODO: Confirm defaults make sense

# DEFAULTS 

# DATASET Config
dir_path = os.getcwd()
DATASET =  "mnist"
DATAPATH = os.path.join(dir_path, "data")

# WEIGHTS Config
WEIGHTS_LR = .01
WEIGHTS_LR_MIN = .00001
WEIGHTS_MOMENTUM = 1.
WEIGHTS_WEIGHT_DECAY = 0.0003
WEIGHTS_GRADIENT_CLIP = 1

# TRAINING CONFIG
EPOCHS = 100
BATCH_SIZE = 100

# ALPHA Optimizer Config
ALPHA_WEIGHT_DECAY = 0.0003
ALPHA_LR = .01

# HDARTS Config
NUM_LEVELS = 2
NUM_NODES_AT_LEVEL = { 0: 3, 1: 3 }
NUM_OPS_AT_LEVEL = { 0: LEN_OPS, 1: 3}
CHANNELS_START = 3
STEM_MULTIPLIER = 1

# MISCELLANEOUS CONFIG
NUM_DOWNLOAD_WORKERS = 2
PRINT_STEP_FREQUENCY = 1
PERCENTAGE_OF_DATA = 100
CHECKPOINT_PATH = os.path.join(dir_path, "checkpoints")

def get_parser(name):
  """ make default formatted parser """
  parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # print default value always
  parser.add_argument = partial(parser.add_argument, help=' ')
  return parser

def parse_gpus(gpus):
    if torch.cuda.is_available():
        if gpus == 'all':
            return list(range(torch.cuda.device_count()))
        else:
            return [int(s) for s in gpus.split(',')]
    else:
        return [0]

class BaseConfig(argparse.Namespace):
  def print_params(self, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(vars(self).items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")

  def as_markdown(self):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(vars(self).items()):
        text += "|{}|{}|  \n".format(attr, value)

    return text



class SearchConfig(BaseConfig):
  def build_parser(self):
    parser = get_parser("Search config")
    parser.add_argument('--name', default='HDARTS')
    parser.add_argument('--datapath', default=DATAPATH)
    parser.add_argument('--dataset', default=DATASET, help='cifar10 / mnist / fashionmnist')
    parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
    parser.add_argument('--stem_multiplier', type=int, default=STEM_MULTIPLIER)
    parser.add_argument('--channels_start', type=int, default=CHANNELS_START)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')

    parser.add_argument('--num_nodes_at_level', type=eval, default=str(NUM_NODES_AT_LEVEL), 
      help='string dictionary detailing nodes at each level i.e.  "{0: 2, 1: 2 }"')
    parser.add_argument('--num_ops_at_level', type=eval, default=str(NUM_OPS_AT_LEVEL), 
      help='string dictionary detailing number of operations at each level i.e.  "{0: 2, 1: 2 }"'),


    parser.add_argument('--alpha_lr', type=float, default=ALPHA_LR, help='lr for weights')
    parser.add_argument('--alpha_weight_decay', type=float, default=ALPHA_WEIGHT_DECAY, #original -> 3e-4,
                        help='weight decay for alpha architecture')

    parser.add_argument('--weights_lr', type=float, default=WEIGHTS_LR, help='lr for weights')
    parser.add_argument('--weights_lr_min', type=float, default=WEIGHTS_LR_MIN, help='minimum lr for weights')
    parser.add_argument('--weights_momentum', type=float, default=WEIGHTS_MOMENTUM, help='momentum for weights')
    parser.add_argument('--weights_weight_decay', type=float, default=WEIGHTS_WEIGHT_DECAY,
                        help='weight decay for weights')
    parser.add_argument('--weights_gradient_clip', type=float, default=WEIGHTS_GRADIENT_CLIP,
                        help='gradient clipping for weights')

    parser.add_argument('--num_download_workers', type=int, default=NUM_DOWNLOAD_WORKERS)
    parser.add_argument('--print_step_frequency', type=int, default=PRINT_STEP_FREQUENCY, help='print frequency')
    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='# of training epochs')
    parser.add_argument('--init_channels', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1, help='# of workers')
    parser.add_argument('--logdir', default="runs", help="directory to write tensorboard logs to. Do not append /.")
    parser.add_argument('--checkpoint_path', default="checkpoints", help="directory to save checkpoints in")
    parser.add_argument('--percentage_of_data', type=int, default=PERCENTAGE_OF_DATA, help="percentage of the dataset to use")
    return parser

  '''
    Description: Turns the lowercase args that are entered into upper case
  '''
  def uppercaseParserArgs(self, args):
    # Make separate list so as not to change
    # size of whats being iterated
    parserArgs = list(vars(args))
    for var_name in parserArgs:
      newVal = getattr(args, var_name)
      setattr(args, var_name.upper(), newVal)


  def __init__(self):
    parser = self.build_parser()

    args = parser.parse_args()
    self.uppercaseParserArgs(args)

    super().__init__(**vars(args))

    self.path = os.path.join('searchs', self.name)
    self.plot_path = os.path.join(self.path, 'plots')
    self.gpus = parse_gpus(self.gpus)
