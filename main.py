import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
from util import addArgs, timer
import util

class HDARTS:

  # Constructor - initializes data loader
  def __init__(self):
    self.dataloader = self.transformImagesToTensors()
    self.model = self.createNetwork(trainloader)

    #Leon return model criterion and optimizer and when training, time that
    self.trainNetwork(trainloader=self.dataloader, model=self.model)
    #self.showPrediction(trainloader=trainloader, model=model)


  """
    Description: Load images and transforms them to tensors
    Args:
      self: instance of class
      trainloader:
    Returns: 

  """
  def transformImagesToTensors(self, imagesetPath='~/.pytorch/MNIST_data/'):
    #Normalize images with a standard deviation of 1
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(imagesetPath, 
      train=True, 
      transform=transform,
      download=True
    )

    #Combine the image and its corresponding label in a package
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return trainloader

  """
    Description: Creates a hardcoded model that will later processes images
    Args:
      self: instance of class
      trainloader:
    Returns: 

  """
  def createNetwork(self, trainloader):
    #print(trainloader.dataset.data.shape)#Will give 64, 1, 28, 28
    #This means we have 64 images, 1 channel(greyscale images), 28 x 28 dimension picture

    #28 x 28 picture = 784, our input size is 784

    #Hidden Layers 
    input_size = trainloader.dataset.data.shape[1] * trainloader.dataset.data.shape[2]
    hidden_layers = [128, 64]
    output_size = 10

    model = nn.Sequential(
      nn.Linear(input_size, hidden_layers[0]),
      nn.ReLU(),
      nn.Linear(hidden_layers[0], hidden_layers[1]),
      nn.ReLU(),
      nn.Linear(hidden_layers[1], output_size),
      nn.LogSoftmax(dim=1)
    )
    return model

  """
    Description:
    Args:
      self: instance of class
      trainloader:
      model: 
    Returns: 
  """
  @util.timer
  def trainNetwork(self, trainloader,  model):
    criterion = nn.NLLLoss()# To compute errors along the way
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    
    #Number of training passes over entire data set
    epochs = 2
    for e in range(epochs):
      running_loss = 0
      for images, labels in trainloader:
        #Flatten the image from 28 x 28 to 784 column vector
        images = images.view(images.shape[0], -1)

        # setting gradient to zeros
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        #Run backward propagation
        loss.backward()

        #update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()

      else: 
        print('Training loss: ', (running_loss/len(trainloader)))

  """

  """
  def showPrediction(self, trainloader, model): 
    # Getting the image to test
    images, labels = next(iter(trainloader))
    # Flatten the image to pass in the model
    img = images[1].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
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

if __name__ == "__main__":
  neuralArchitectureSearcher = HDARTS()


