""" Config class for search/augment """
import argparse
import os
#import genotypes as gt
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch





from functools import wraps
from time import time
def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

# Get number of edges in complete dag with n nodes
def num_edges_for_dag(n: int):
    if ((n - 1) % 2 == 0):
        return int((n - 1) / 2) * n
    else: 
        return int(n/2) * (n - 1)


"""
    Description: 
    Args: 
        trainloader: PyTorch formatted loader
        model: PyTorch formatted model
    Returns:
"""
def showPrediction(trainloader, model): 
  # Getting the image to test
  images, labels = next(iter(trainloader))
  # Flatten the image to pass in the model
  img = images[1].view(1, 784)
  # Turn off gradients to speed up this part
  with torch.no_grad():
      logps = model(img)
  # Output of the network are log-probabilities, 
  # need to take exponential for probabilities
  ps = torch.exp(logps)
  view_classify(img, ps)



  
def view_classify(img, ps):
  ps = ps.data.numpy().squeeze()
  fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
  ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
  ax1.axis('off')
  ax2.barh(np.arange(10), ps)
  ax2.set_aspect(0.1)
  ax2.set_yticks(np.arange(10))
  ax2.set_yticklabels(np.arange(10))
  ax2.set_title('Class Probability')
  ax2.set_xlim(0, 1.1)
  plt.tight_layout()
  plt.show()



def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


def addArgs():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTSSSSS', help='which architecture to use')
    args = parser.parse_args()
    print('\n\n' + str(args) + '\n\n')