# Internal Imports
from alpha import Alpha
from learnt_model import LearntModel
from operations import OPS
from util import load_alpha

# External Imports
from copy import deepcopy
from lucent.modelzoo import inceptionv1
from lucent.optvis import render
from model import Model
from pprint import pprint
import random
import sys
import torch
import torch 
import torch.nn as nn
import torch.nn.functional as f

# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Constants
NUM_EPOCHS = 50

# Load model from argument
model = torch.load("checkpoints_train/cifar10/21-04-2021--17-10-44/best.pt")
pprint("Model")
pprint(model)
'''
loss_criterion = torch.nn.BCELoss()
x = nn.Parameter(torch.rand(1, 1, 2, 2))
params = nn.ParameterList([x])
optim = torch.optim.Adam(
            params=params,
            lr=0.001,
            betas=(0.5, 0.999),
            weight_decay=1e-3)
desired_activation = torch.ones(1, 1, 2, 2)
for epoch in range(NUM_EPOCHS):
    activation = model.forward(x, x, module_outputs_to_get=[(0, 2)])[(0,2)]
    optim.zero_grad()
    loss = -torch.dot(activation[0, :, 0, 0], desired_activation[0, :, 0, 0])
    loss.backward()
    optim.step()
    print("Epoch:", epoch, " loss = ", loss.item())

print("Final Input Image", x)
print("Activation Achieved", model.forward(x, x, module_outputs_to_get=[(0, 2)])[0,2])
'''

model.eval()
def objective_function(model, img):
    _, module_outputs = model.forward(img, module_outputs_to_get={0: [(0,2)]})
    return -module_outputs[0][(0,2)].mean()
render.render_vis(model, objective_function)
