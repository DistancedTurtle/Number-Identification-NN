import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transfrom.ToTensor])

train = torchvision.datasets.MNIST(root='./data', train=True, download=true, transform=transform)
test = torchvision.datasets.MNIST(root='./data', train=False, download=true, transform=transform)