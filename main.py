import numpy as np
import torchvision
from torchvision import datasets, transforms
import utils

transform = transforms.Compose([transforms.ToTensor])

train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train.data.numpy()
X_test = test.data.numpy()

y_train = train.targets.numpy()
y_test = test.targets.numpy()

utils.print_image_grid(X_train,0)

X_train = utils.normalize(X_train)
X_test = utils.normalize(X_test)

utils.print_image_grid(X_train,0)