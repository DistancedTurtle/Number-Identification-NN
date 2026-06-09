import numpy as np

def normalize(array: np.ndarray) -> np.ndarray[float]: 
    return(array/255.0)

def print_image_grid(array: np.ndarray, image_index: int):
    np.set_printoptions(linewidth=200, threshold=np.inf)
    print(array[image_index,:,:])