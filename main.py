import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transfrom.ToTensor])

train = torchvision.datasets.MNIST(root='./data', train=True, download=true, transform=transform)
test = torchvision.datasets.MNIST(root='./data', train=False, download=true, transform=transform)

X_train = train.data.numpy()
X_test = test.data.numpy()

y_train = train.targets.numpy()
y_test = test.targets.numpy()

print(f"Train images: {X_train.shape}, type: {type(X_train)}")
print(f"Train labels: {y_train.shape}, type: {type(y_train)}")

