import numpy as np

def normalize(array: np.ndarray) -> np.ndarray[float]: 
    return(array/255.0)

def print_image_grid(array: np.ndarray, image_index: int):
    np.set_printoptions(linewidth=200, threshold=np.inf)
    print(array[image_index,:,:])

def ReLU(arr: np.ndarray) -> np.ndarray:
    arr.clip(min=0, out=arr)
    return(arr)

def softmax(arr):
    arr = np.exp(arr)/(np.sum(np.exp(arr), axis = 1, keepdims=True))
    return(arr)