import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

import hdarts

class HDARTS:

  # Constructor - initializes data loader
  def __init__(self):
    self.runMNISTTraining()


  """
  Description: Runs MNIST training and prediction
    Args:
      self: instance of class
    Returns: 
  """
  def runMNISTTraining(self):
    self.dataloaderMNIST = self.transformImagesToTensors()
    self.modelMNIST = self.createModel(self.dataloaderMNIST)

    #Leon return model criterion and optimizer and when training, time that
    self.trainNetwork(trainloader=self.dataloaderMNIST, model=self.modelMNIST)
    hdarts.util.showPrediction(trainloader=self.dataloaderMNIST, model=self.modelMNIST)



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
  def createModel(self, trainloader):
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
  Description: Downloads CIFAR10 and stores it in './data' and
    if you already have it it just loads it
  Args: 

  Returns:
    trainloader: 
    testloader:
  """
  def transformImagesToTensorsCIFAR10():
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
      download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
      shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
      download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
      shuffle=False, num_workers=2)


    return trainloader, testloader

  """
    Description:
    Args:
      self: instance of class
      trainloader:
      model: 
    Returns: 
  """
  @hdarts.util.timer
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



if __name__ == "__main__":
  neuralArchitectureSearcher = HDARTS()


