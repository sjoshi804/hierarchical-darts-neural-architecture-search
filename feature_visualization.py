# Internal Imports
from torch.optim import optimizer
from operations import OPS
from alpha import Alpha
from hierarchical_operation import HierarchicalOperation

# External Imports
from model import Model
import torch 
import torch.nn as nn
import torch.nn.functional as f
import sys
from pprint import pprint
import random

# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Constants
NUM_EPOCHS = 50

# Load model from argument
alpha = Alpha(2, {0: 3, 1: 4}, {0: 1, 1: 1})
model = HierarchicalOperation.create_dag(1, alpha, alpha.parameters[1], OPS, 2,2,2, learnt_op=True)
baseline = torch.rand(1, 2, 2, 2)

pprint("Model")
pprint(model)

loss_criterion = torch.nn.BCELoss()

x = torch.rand(1, 2, 2, 2)
params = nn.ParameterList()
for image in x:
    for channel in image:
        for col in channel:
            for elem in col:
                elem = nn.Parameter(elem)
                params.append(elem)
print(x)
optim = torch.optim.Adam(
            params=params,
            lr=0.01,
            betas=(0.5, 0.999),
            weight_decay=1e-3)
desired_activation = torch.ones(1, 2, 2, 2)
for epoch in range(NUM_EPOCHS):
    activation = model.forward(x, x, module_outputs_to_get=[(0, 2)])[(0,2)]
    optim.zero_grad()
    loss = torch.abs(activation.mean())
    loss.backward()
    print([param.grad for param in params])
    optim.step()
    print("Epoch:", epoch, " loss = ", loss)

print("Final Input Image", x)
print("Activation Achieved", model.forward(x, x, module_outputs_to_get=[(0, 2)])[0,2])