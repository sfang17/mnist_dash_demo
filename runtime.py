import numpy as np
from scipy import signal

# cleans and preps data from drawn input
def generate_input_from_drawing(a):
    # resize the input image
    def shrink(data, rows, cols):
        row_sp = int(a.shape[0]/rows)
        col_sp = int(a.shape[1]/cols)
        _ = np.sum(data[i::row_sp] for i in  range(0, row_sp))
        return np.sum(_[:,i::col_sp] for i in range(0, col_sp))
    
    # add blur to liken input data
    def blur(a):
        kernel = signal.gaussian(2, std=1) # Here you would insert your actual kernel of any size
        a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, a)
        a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, a)
        return a
    
    # performing the processing
    a = a.astype(float) # convert from bool
    a = shrink(a, 28, 28) # resize
    a = blur(a) # add gaussian blur
    a = a/a.max() # minmax scale
    a = np.append(a.sum(axis=0), a.sum(axis=1)) # center
    a = a.reshape([1,-1]) # reshape
    
    return a
