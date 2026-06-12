import numpy as np
import utils
import sys  
from torchvision.datasets import MNIST

#Hyperparameters
EPOCHS = 2000
LEARNING_RATE = .5
HIDDEN_SIZE = 64

def main():
    #Load Data
    train = MNIST(root='./data', train=True, download=True)
    test = MNIST(root='./data', train=False, download=True)

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

    b_1=np.zeros((1,HIDDEN_SIZE))
    b_2=np.zeros((1,10))

    w_1=np.random.randn(784,HIDDEN_SIZE) * .01
    w_2=np.random.randn(HIDDEN_SIZE,10) * .01

    m = X_train.shape[0]

    y_train_oh = identity_matrix[y_train]

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Skipping training. Loading saved weights for testing...")
        try:
            saved_data = np.load("mnist_weights.npz")
            w_1 = saved_data["w_1"]
            b_1 = saved_data["b_1"]
            w_2 = saved_data["w_2"]
            b_2 = saved_data["b_2"]
            print("Weights loaded successfully!")
        except FileNotFoundError:
            print("Error: 'mnist_weights.npz' not found. Please train the model first.")
            sys.exit(1)
            
    else:
        for epoch in range(EPOCHS):
            #hidden layers with ReLU and softmax activation for classification
            h1 = utils.ReLU(np.matmul(X_train,w_1) + b_1)
            h2 = utils.softmax(np.matmul(h1,w_2) + b_2)

            predictions = np.argmax(h2, axis=1)

            #Categorical Cross Entropy loss calculation 
            loss = -1 * np.sum(y_train_oh * np.log(h2 + 1e-15)) / m

            #gauge for loss and model accuracy
            accuracy = np.mean(predictions == y_train) * 100
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} -> Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

            #calculate derivatives for backpropagation
            dz2 = h2 - y_train_oh
            dw2 = np.matmul(h1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            dz1 = np.matmul(dz2, w_2.T) 
            dz1[h1 <= 0] = 0  
            dw1 = np.matmul(X_train.T, dz1) / m
            db1 = np.sum(dz1, axis = 0, keepdims = True) / m

            #update weights and biases
            w_1 -= LEARNING_RATE * dw1
            b_1 -= LEARNING_RATE * db1
            w_2 -= LEARNING_RATE * dw2
            b_2 -= LEARNING_RATE * db2

        np.savez("mnist_weights.npz", w_1=w_1, b_1=b_1, w_2=w_2, b_2=b_2)

    utils.test_model(w_1, w_2, b_1, b_2, X_test, y_test)

if __name__ == "__main__":
    main()