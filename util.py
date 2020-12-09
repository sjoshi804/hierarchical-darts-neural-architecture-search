'''
get_data, Average_Meter: Borrowed from https://github.com/khanrc/pt.darts
'''

# External Imports
from alpha import Alpha
from functools import wraps
from model_controller import ModelController
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import torch
import torchvision.datasets as dset

# Internal Imports
import preProcess


def get_data(dataset_name, data_path, cutout_length, validation):
    """ Get torchvision dataset """

    # If dataset is supported, initialize from torchvision.datasets
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        dataset = dset.MNIST
        n_classes = 10
    else:
        raise ValueError('Unexpected Dataset = {}'.format(dataset_name))

    # Preprocess data - gets the tranformations and then downloads them, torch then applies them by specifying as paramter to dataset
    trn_transform, val_transform = preProcess.data_transforms(dataset_name, cutout_length)
    trn_data = dataset(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dataset(root=data_path, train=False, download=True, transform=val_transform))

    return ret

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model: ModelController, epoch: int, checkpoint_root_dir, is_best=False):
    '''
    Saves alphs and weights to be able to recreate model as is.
    '''
    # Creates checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_root_dir):
        os.makedirs(checkpoint_root_dir)

    # Create directory for this checkpoint 
    current_checkpoint_dir = os.path.join(checkpoint_root_dir, str(epoch))
    if not os.path.exists(current_checkpoint_dir):
        os.makedirs(current_checkpoint_dir)

    # Function to save object to file
    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    # Saves alpha and weights to aforementioned directory
    alpha_file_path = os.path.join(current_checkpoint_dir, "alpha.pkl")
    weights_file_path = os.path.join(current_checkpoint_dir, "weights.pkl")
    save_object(model.alpha, alpha_file_path)
    torch.save(model.model.state_dict(), weights_file_path)

    # Ensure best checkpoint directory exists
    best_checkpoint_dir = os.path.join(checkpoint_root_dir, "best")
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir)

    # If best copies over to best checkpoint directory
    if is_best:
        shutil.copyfile(alpha_file_path, os.path.join(best_checkpoint_dir, "alpha.pkl"))
        shutil.copyfile(weights_file_path, os.path.join(best_checkpoint_dir, "weights.pkl"))
    
    # TODO: Save other parameters needed for ModelController constructor

def load_checkpoint(checkpoint_root_dir, epoch=-1):
    '''
    Creates model from saved files in checkpoint root directory
    If epoch not specified, loads best checkpoint
    '''
    alpha_file_path = "alpha.pkl"
    weights_file_path = "weights.pkl"
    
    # If epoch specified set checkpoint dir to that epoch
    checkpoint_dir = "best"
    if epoch > 0:
        checkpoint_dir = str(epoch)
    alpha_file_path = os.path.join(checkpoint_root_dir, checkpoint_dir, alpha_file_path)
    weights_file_path = os.path.join(checkpoint_root_dir, checkpoint_dir, weights_file_path)

    # Function to load object from file
    def load_object(filename):
        with open(filename, 'rb') as input:
            obj = pickle.load(input)
            return obj
    
    # Gets alpha and weights
    alpha = load_object(alpha_file_path)
    weights = torch.load(weights_file_path)

    return alpha, weights
    #TODO: Model creation from alpha, weights and other parameters

    

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

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


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]
