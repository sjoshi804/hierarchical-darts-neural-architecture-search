import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



"""
Description: Trains the Alpha Parameters of the NN


"""
class AlphaTrainer:
    def __init__(
        self, 
        model, #Neural network we will output. Changing image weights 
        args
    ):
        #Momentum: average of previous gradients to reduce variants
        self.network_momentum = args.momentum
        #Reduces complexity by adding sum of weights to decay constants
        self.network_weight_decay = args.weight_decay
        self.model = model

        #Adam optimizer: 
        # 1) Has a learning rate for each paramter
        # 2) Has a learning rate for per-parameter learning based on
        #   average of recent magnitudes of previous gradients
        #This is where we specify our loss and gradients are taken wrt to architecture parameters
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, 
            betas=(0.5, 0.999), 
            weight_decay=args.arch_weight_decay
        )

    #None of this makes sense
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        #Since PyTorch accumulates gradients on each backward pass we have to zero it out
        #each time
        self.optimizer.zero_grad()

        #This does something to the gradients, but what??

        #Leave this here. If this runs we need 
        #to understand "_backward_step_unrolled"
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            #Taking the simplest model of approximations
            #From Sid: Finding errors for backpropagation on the validation set
            #Here it makes sense its calculating the error for the whole architecture
            #because we are in architect.py and computing error against the
            #validation data 
            self._backward_step(input_valid, target_valid)

        #??This performs a parameter(weight??) update on the current gradient,whatever that means
        self.optimizer.step()

    """

    """
    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
