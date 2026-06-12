import numpy as np

def normalize(array: np.ndarray) -> np.ndarray: 
    return(array/255.0)

def print_image_grid(array: np.ndarray, image_index: int):
    np.set_printoptions(linewidth=200, threshold=np.inf)
    print(array[image_index,:,:])

def ReLU(arr: np.ndarray) -> np.ndarray:
    return np.maximum(0, arr)

def softmax(arr):
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

def test_model(w_1, w_2, b_1, b_2, X_test, y_test):

    h1 = ReLU(np.matmul(X_test,w_1) + b_1)
    h2 = softmax(np.matmul(h1,w_2) + b_2)

    predictions = np.argmax(h2, axis=1)

    accuracy = np.mean(predictions == y_test) * 100

    print(f"model accuracy: {accuracy}%")