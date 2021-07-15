# External Imports
from typing import Dict 
import torch
import torch.nn as nn 

# Internal Imports
from alpha import Alpha 
from model import Model

class ModelController(nn.Module):
    '''
    This class is the controller for model and has alpha parameters registered in addition to the weights (weights) parameters automatically registered by Pytorch.

    get_weights -> returns weights parameters

    get_alpha_level(level) -> returns parameter (yes singular, as the whole tensor is wrapped as one parameter) corresponding to alpha_level
    '''
    def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int], primitives: dict, channels_in: int, channels_start: int, stem_multiplier: int,  num_classes: int, loss_criterion, num_cells: int, writer=None, test_mode=False):
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
        self.channels_start = channels_start
        self.stem_multiplier = stem_multiplier
        self.num_classes = num_classes
        self.loss_criterion = loss_criterion
        self.writer = writer 
        self.num_cells = num_cells
        self.test_mode = test_mode
        self.graph_added = False

        # Initialize Alpha for both types of cells
        # Normal Cell
        self.alpha_normal = Alpha(
            num_levels=self.num_levels,
            num_nodes_at_level=self.num_nodes_at_level,
            num_ops_at_level=self.num_ops_at_level,
            randomize=True
        )

        self.alpha_reduce = Alpha(
            num_levels=self.num_levels,
            num_nodes_at_level=self.num_nodes_at_level,
            num_ops_at_level=self.num_ops_at_level,
            randomize=True
        )

        # Initialize model with initial alpha
        self.model = Model(
                alpha_normal=self.alpha_normal,
                alpha_reduce=self.alpha_reduce,
                primitives=self.primitives,
                channels_in=self.channels_in,
                channels_start=self.channels_start,
                stem_multiplier=self.stem_multiplier,
                num_classes=self.num_classes,
                num_cells=num_cells,
                writer=writer,
                test_mode=test_mode)
        
        if not test_mode and torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, x, temp=None):
        if self.test_mode and not self.graph_added: 
            # Visualize model in tensorboard
            self.writer.add_graph(self.model, x)
            self.graph_added = True
        return self.model(x, temp=temp)

    # Get loss object using loss_criterion
    def loss(self, X, y):
        logits = self.forward(X)
        return self.loss_criterion(logits, y)

    # Get list of alpha parameters for a level
    def get_alpha_level(self, level):
        return self.alpha_normal.get_alpha_level(level).extend(self.alpha_reduce.get_alpha_level(level))

    # Get all the weights parameters
    def get_weights(self):
        weights = nn.ParameterList()
        for name, param in self.named_parameters(recurse=True):
            if 'alpha' not in name:
                weights.append(param)
        return weights
    
    # Sets requires grad to false for alpha params / true for weight params
    def weight_training_mode(self):
        for name, param in self.named_parameters(recurse=True):
            if 'alpha' in name:
                param.requires_grad = False 
            else:
                param.requires_grad = True
    
    # Sets requires grad to false for weight params / true for alpha params
    def alpha_training_mode(self):
        for name, param in self.named_parameters(recurse=True):
            if 'alpha' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # Assumes in alpha training mode overall i.e. weight gradients are turned off
    # Switches gradient off for all other levels
    def alpha_training_mode_for_level(self, level):
        for i in range(self.alpha_normal.num_levels):
            for param in self.alpha_normal.get_alpha_level(i):
                if i == level:
                    param.requires_grad = True 
                else: 
                    param.requires_grad = False