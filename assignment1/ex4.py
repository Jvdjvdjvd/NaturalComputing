import random
import numpy as np
import matplotlib.pyplot as plt

#Set global paramameters
pm = 0.2
genLen = 10

def bitflipper(original_vector, flip_chance):
    idxes = [i for i in range(len(original_vector)) if random.random() < flip_chance]
    subtracter = np.zeros(len(original_vector)).astype(int)
    subtracter[idxes] = 1
    xm = np.absolute(original_vector - subtracter)
    return xm

def get_fitness(x): #TODO: check if this is indeed the fitness function..

    return sum(x)

def do_genetic_run(length, iterations, pm, alwaysReplace = False):
    x = np.random.randint(2, size=length)
    xList = [[] for _ in range(iterations+1)]
    fitnessList = [0 for _ in range(iterations+1)]
    fitnessList[0] = get_fitness(x)
    xList[0] = x

    for i in range(1,iterations+1):
        xList[i] = x
        xm = bitflipper(x, pm)
        if get_fitness(xm) > get_fitness(x) or alwaysReplace:
            x = xm
        xList[i] = x
        fitnessList[i] = get_fitness(x)
    return xList, fitnessList

if __name__ == '__main__':
    # question 4a
    Xs,Fitnesses = do_genetic_run(100,1500,1/100)
    plt.plot(list(range(0,1501)), Fitnesses)

    #4b
    plt.figure()
    for i in range(10):
        Xs, Fitnesses = do_genetic_run(100, 1500, 1 / 100)
        plt.plot(list(range(0, 1501)), Fitnesses)

    #4c
    Xs,Fitnesses = do_genetic_run(100,1500,1/100, alwaysReplace=True)
    plt.plot(list(range(0,1501)), Fitnesses)


#TODO: check if the questions are actually answered
