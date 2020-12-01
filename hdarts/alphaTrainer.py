from model import Model
import torch
import torch.nn as nn



"""
Description: Trains the Alpha Parameters of the NN
"""

# TODO: Needs to know loss function in terms of alpha parameters and input
# TODO: What about designing a model controller like pt.darts?
class AlphaTrainer:
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = Model(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):
        return self.net(x, weights_normal, weights_reduce)