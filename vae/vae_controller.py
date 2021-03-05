# External Imports
from typing import Dict 
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Internal Imports
from alpha import Alpha 
from .beta_vae import BetaVAE

class VAEController(nn.Module):
    '''
    This class is the controller for VAE and has alpha parameters registered in addition to the weights (weights) parameters automatically registered by Pytorch.

    get_weights -> returns weights parameters

    get_alpha_level(level) -> returns parameter (yes singular, as the whole tensor is wrapped as one parameter) corresponding to alpha_level
    '''
    def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int], primitives: dict, channels_in: int, beta: float, image_height: int, image_width: int, writer=None, test_mode=False):
        '''
        - Initializes member variables
        - Registers alpha parameters by creating a dummy alpha using the constructor and using get_alpha_level to get the alpha for a given level. This tensor is wrapped with nn.Parameter to indicate that is a Parameter for this controller (thus requires gradient computation with respect to itself). This nn.Parameter is added to the nn.ParameterList that is self.alphas.
        - Registers weights parameters by creating a model from aforementioned dummy alpha
        '''
        # Superclass constructor
        super().__init__()

        # Initialize member variables
        self.num_levels = num_levels
        self.num_nodes_at_level = num_nodes_at_level
        self.num_ops_at_level = num_ops_at_level
        self.primitives = primitives
        self.channels_in = channels_in
        self.beta = beta
        self.writer = writer 

        # Initialize Alpha
        self.alpha = Alpha(
            num_levels=self.num_levels,
            num_nodes_at_level=self.num_nodes_at_level,
            num_ops_at_level=self.num_ops_at_level
        )

        # Initialize model with initial alpha, 
        self.vae = BetaVAE(
                alpha=self.alpha,
                beta=beta,
                primitives=self.primitives,
                channels_in=self.channels_in,
                image_height=image_height,
                image_width=image_width,
                writer=writer,
                test_mode=test_mode)
        
        if not test_mode and torch.cuda.is_available():
            self.vae = self.vae.cuda()

    def forward(self, x):
        return self.vae(x)

    def loss(self, x, output):
        return self.vae.loss(x, output)

    def disentanglement(self, x, output):
        return self.vae.disentanglement(x, output)

    # Get list of alpha parameters for a level
    def get_alpha_level(self, level):
        return self.alpha.get_alpha_level(level)

    # Get all the weights parameters
    def get_weights(self):
        weights = nn.ParameterList()
        for name, param in self.named_parameters(recurse=True):
            if 'alpha' not in name:
                weights.append(param)
        return weights