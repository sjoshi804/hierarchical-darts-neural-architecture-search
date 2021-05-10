'''
Borrowed from https://github.com/khanrc/pt.darts
'''

import numpy as np
import torch
import torchvision.transforms as transforms

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

# Specifies the transformations to be applied to the data
def data_transforms(dataset, cutout_length):

    dataset = dataset.lower()
    if dataset == 'cifar10':
        return _data_transforms_cifar10()
    elif dataset == 'mnist':
        # Initialized to avoid expensive recomputation
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        
        # Determine randomAffine to avoid learning dependency on unwanted specifics such as position, orientation etc.
        randomAffine = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    else:
        raise ValueError('Unexpected Dataset = {}'.format(dataset))

    # Normalization - first needs to convert to tensors
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    # Apply random affine only to training data
    train_transform = transforms.Compose(randomAffine + normalize)
    valid_transform = transforms.Compose(normalize)

    
    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform

'''
Borrowowed from Quark0/DARTS
'''
def _data_transforms_cifar10():
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  
  train_transform.transforms.append(Cutout(16))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform