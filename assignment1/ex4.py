import random
import numpy as np

#Set global paramameters
pm = 0.2
genLen = 10

def bitflipper(original_vector, flip_chance):
    idxes = [i for i in range(len(original_vector)) if random.random() < flip_chance]
    subtracter = np.zeros(len(original_vector)).astype(int)
    subtracter[idxes] = 1
    xm = np.absolute(x - subtracter)
    return xm


#start ex4
x = np.random.randint(2, size=genLen)
xm = bitflipper(x, pm)

#TODO: implement the rest of question 4