# External Imports
from datetime import datetime
from torch import tensor 
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn
import unittest

# Internal Imports 
from model_controller import ModelController
from operations import LEN_SIMPLE_OPS, SIMPLE_OPS

class TestModelController(unittest.TestCase):
  
  def test_times_two_function(self):
    # Hyperparameters
    num_levels = 2 
    num_nodes_at_level = {0: 2, 1: 2}
    num_ops_at_level = {0: LEN_SIMPLE_OPS, 1: 1}
    num_epochs = 100

    # Initialize tensorboard writer
    dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    writer = SummaryWriter('test/test_double_func/' + str(dt_string) + "/")
  
    # Define model 
    model = ModelController(
            num_levels=num_levels,
            num_nodes_at_level=num_nodes_at_level,
            num_ops_at_level=num_ops_at_level,
            primitives=SIMPLE_OPS,
            channels_in=1,
            channels_start=1,
            stem_multiplier=1,
            num_classes=1,
            loss_criterion=nn.L1Loss(),
            writer=writer,
            test_mode=True
        )

    # Input
    x = tensor([
    [
      # feature 1
      [
        [1.]
      ]
    ]
    ])

    # Expected output
    y = tensor([
    [
      # feature 1
      [
        [2.]
      ]
    ]
    ])

    # Alpha Optimizer - one for each level
    alpha_optim = []
    for level in range(0, num_levels):
      alpha_optim.append(torch.optim.Adam(
                params=model.get_alpha_level(level),
                lr=0.1))

    for _ in range(0, num_epochs):
      # Alpha Gradient Steps for each level
      for level in range(0, num_levels):
        alpha_optim[level].zero_grad()
        loss = model.loss_criterion(model(x), y)
        print(loss)
        loss.backward()
        alpha_optim[level].step()
    
if __name__ == '__main__':
  unittest.main()