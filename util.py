'''
get_data, Average_Meter: Borrowed from https://github.com/khanrc/pt.darts
'''

# External Imports
import csv
import io
from functools import wraps
from time import time
import math
import numpy as np
import os
import pickle
import shutil
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.datasets as dset


# Internal Imports
from alpha import Alpha
import preProcess


def get_data(dataset_name, data_path, cutout_length, validation):
    """ Get torchvision dataset """

    # If dataset is supported, initialize from torchvision.datasets
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        dataset = dset.MNIST
        n_classes = 10
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10
        n_classes = 10
    else:
        raise ValueError('Unexpected Dataset = {}'.format(dataset_name))

    # Preprocess data - gets the tranformations and then downloads them, torch then applies them by specifying as paramter to dataset
    trn_transform, val_transform = preProcess.data_transforms(dataset_name, cutout_length)
    trn_data = dataset(root=data_path, train=True, download=True, transform=trn_transform)


    if dataset_name == 'mnist':
        shape = trn_data.data.shape
    elif dataset_name == 'cifar10':
        shape = trn_data.data.shape
            

    # assuming shape is NHW or NHWC
    # shape = trn_data.train_data.shape
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

def print_alpha(alpha: Alpha):
    for level in alpha.parameters:
        print("Level", level+1)
        for op_num in range(0, len(alpha.parameters[level])):
            print("Operation", op_num)
            for edge in alpha.parameters[level][op_num]:
                print(edge, int(np.argmax(alpha.parameters[level][op_num][edge].cpu().detach())), alpha.parameters[level][op_num][edge])
            print("")
        print("\n")

def create_alpha_history_object(alpha):
    # Initialize alpha_i for all i < num_levels
    alpha_history = {}
    for i in range(0, alpha.num_levels):
        alpha_i = [ {}  for y in range(0, alpha.num_ops_at_level[i+1])]
        for dict in alpha_i:
            for node_a in range(0, alpha.num_nodes_at_level[i]):
                for node_b in range(node_a + 1, alpha.num_nodes_at_level[i]):
                    # Skip creation of alpha if top level and edges don't exist
                    if (i == alpha.num_levels - 1) and(node_a < 2) and ((node_b == 1) or (node_b == alpha.num_nodes_at_level[i] - 1)):
                        continue

                    # Initialize a list to track the chosen operation at each stage
                    dict[(node_a, node_b)] = []
        # Add level to history object
        alpha_history[i] = alpha_i
    return alpha_history

def update_alpha_history(alpha_history, alpha):
    for level in alpha.parameters:
        for op_num in range(0, len(alpha.parameters[level])):
            for edge in alpha.parameters[level][op_num]:
                chosen_op = int(np.argmax(alpha.parameters[level][op_num][edge].cpu().detach()))
                alpha_history[level][op_num][edge].append(chosen_op)
    return alpha_history

def write_alpha_history_to_csvs(alpha_history, alpha, alpha_type, write_dir):
    # Creates checkpoint directory if it doesn't exist                                                            
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    curr_dir = os.getcwd()
    os.chdir(write_dir)
    for level in alpha.parameters:
        for op_num in range(0, len(alpha.parameters[level])):
            for edge in alpha.parameters[level][op_num]:
                with open(alpha_type + "-level-" + str(level) + "-op-" + str(op_num) + "-edge-" + str(edge) + ".csv", mode='w+') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for epoch, chosen_op in enumerate(alpha_history[level][op_num][edge]):
                        csv_writer.writerow([epoch, chosen_op])
    os.chdir(curr_dir)
    return 

def print_alpha_tensorboard(alpha: Alpha, writer: SummaryWriter, type: str, epoch=None):

    # Write to temp file for easy parsing
    with open('final_alpha.txt', 'w+') as f:
        # Reroute stdout to file
        sys.stdout = f

        print("Epoch", str(epoch))
        for level in alpha.parameters:
            print("Level", level)
            for op_num in range(0, len(alpha.parameters[level])):
                print("Operation", op_num)
                for edge in alpha.parameters[level][op_num]:
                    print(edge, alpha.parameters[level][op_num][edge])
                print("")
            print("\n")
    
    # Read from file and print to tensorboard for results analysis
    with open('final_alpha.txt', 'r') as f:
        writer.add_text("Alpha at epoch: " + str(epoch), f.read())

def save_checkpoint(model, epoch: int, checkpoint_root_dir, is_best=False):
    '''
    Saves alpha to be able to learn the model from here if everything crashes.
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
    alpha_normal_file_path = os.path.join(current_checkpoint_dir, "alpha_normal.pkl")
    alpha_reduce_file_path = os.path.join(current_checkpoint_dir, "alpha_reduce.pkl")
    save_object(model.alpha_normal, alpha_normal_file_path)
    save_object(model.alpha_reduce, alpha_reduce_file_path)

    # Ensure best checkpoint directory exists
    best_checkpoint_dir = os.path.join(checkpoint_root_dir, "best")
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir)

    # If best copies over to best checkpoint directory
    if is_best:
        shutil.copyfile(alpha_normal_file_path, os.path.join(best_checkpoint_dir, "alpha_normal.pkl"))
        shutil.copyfile(alpha_reduce_file_path, os.path.join(best_checkpoint_dir, "alpha_reduce.pkl"))
        # shutil.copyfile(weights_file_path, os.path.join(best_checkpoint_dir, "weights.pkl"))

# Function to load object from file
def load_object(filename):
    with open(filename, 'rb') as input:
        if torch.cuda.is_available():
            return pickle.load(input)
        else:
            return CPU_Unpickler(input).load()

def load_alpha(alpha_dir_path, epoch=None):
    if epoch is None:
        epoch = "best"
    alpha_normal = load_object(os.path.join(alpha_dir_path, str(epoch), "alpha_normal.pkl"))
    alpha_reduce = load_object(os.path.join(alpha_dir_path, str(epoch), "alpha_reduce.pkl"))
    return alpha_normal, alpha_reduce

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
    
    # Gets alpha and weights
    alpha = load_object(alpha_file_path)
    weights = torch.load(weights_file_path)

    return alpha, weights

    

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


def parse_gpus(gpus):
    if torch.cuda.is_available():
        if gpus == 'all':
            return list(range(torch.cuda.device_count()))
        else:
            return [int(s) for s in gpus.split(',')]
    else:
        return [0]

def det_cell_size(num_darts_nodes: int):
    num_darts_nodes += 1 # Since output node counts as node for us
    num_ops = {}
    def binom(n, k):
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    num_ops_darts = binom(num_darts_nodes, 2) - 3
    for n in range(3, num_darts_nodes):
        for m in range(4, num_darts_nodes):
            num_ops[(n, m)] = (binom(m, 2) - 3) * binom(n, 2)
    sorted_keys = sorted(num_ops.keys(), key=lambda x: abs(num_ops_darts-num_ops[x]))
    print("DARTS ops", num_ops_darts)
    print("Closest HDARTS Candidates")
    for i in range(3):
        print("Level 0", sorted_keys[i][0], "Level 1", sorted_keys[i][1], "Num Ops", num_ops[sorted_keys[i]])
    return sorted_keys
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob).mul_(mask)
  return x
