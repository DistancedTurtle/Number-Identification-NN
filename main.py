import numpy as np
import torchvision
from torchvision import datasets, transforms
import utils

transform = transforms.Compose([transforms.ToTensor])

#load data
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train.data.numpy()
X_test = test.data.numpy()

y_train = train.targets.numpy()
y_test = test.targets.numpy()

X_train = utils.normalize(X_train)
X_test = utils.normalize(X_test)

#flatten data 
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1,784)

#one hot encode y_train
identity_matrix = np.eye(10)

'''explanation:
y_train = np.array([2, 0, 1])
identity_matrix = np.eye(3)
Row 0 -> [1, 0, 0]
Row 1 -> [0, 1, 0]
Row 2 -> [0, 0, 1]
numpy takes the values of the array(y_train in this case) and then uses those 
values as indexes and selects those rows from the indexed matrix to be added to new array
[
 [0, 0, 1],  # Represents the original '2'
 [1, 0, 0],  # Represents the original '0'
 [0, 1, 0]   # Represents the original '1'
]'''

y_train_oh = identity_matrix[y_train]



'''print checks for shape:
print("X_train shape:", X_train.shape)       # Should be (60000, 784)
print("y_train_oh shape:", y_train_oh.shape) # Should be (60000, 10)
print("X_test shape:", X_test.shape)         # Should be (10000, 784)
print("y_test shape:", y_test.shape)         # Should be (10000,)'''

b_1=np.zeros((1,64))
b_2=np.zeros((1,10))

w_1=np.random.randn(784,64) * .01
w_2=np.random.randn(64,10) * .01
epochs = 1000
learning_rate = .5
m = X_train.shape[0]
for epoch in range(epochs):
    h1 = utils.ReLU(np.matmul(X_train,w_1) + b_1)
    h2 = utils.softmax(np.matmul(h1,w_2) + b_2)

    predictions = np.argmax(h2, axis=1)

    #Categorical Cross Entropy loss calculation 

    loss = -1 * np.sum(y_train_oh * np.log(h2 + 1e-15)) / m

    #efficency gauge
    accuracy = np.mean(predictions == y_train) * 100
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} -> Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

    dz2 = h2 - y_train_oh
    dw2 = np.matmul(h1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.matmul(dz2, w_2.T) 
    dz1[h1 <= 0] = 0  
    dw1 = np.matmul(X_train.T, dz1) / m
    db1 = np.sum(dz1, axis = 0, keepdims = True) / m

    w_1 -= learning_rate * dw1
    b_1 -= learning_rate * db1
    w_2 -= learning_rate * dw2
    b_2 -= learning_rate * db2

np.savez("mnist_weights.npz", w_1=w_1, b_1=b_1, w_2=w_2, b_2=b_2)
