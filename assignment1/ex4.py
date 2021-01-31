#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt


def bitflipper(original_vector, flip_chance):
    """
    Flip each bit in the given vector with the given chance.
    """

    idxes = [i for i in range(len(original_vector)) if random.random() < flip_chance]
    subtracter = np.zeros(len(original_vector)).astype(int)
    subtracter[idxes] = 1
    return np.absolute(original_vector - subtracter)


def get_fitness(x):
    """
    Compute the fitness of the given candidate.
    """

    return sum(x)


def do_genetic_run(length, iterations, pm, alwaysReplace=False):
    """
    Execute a single GA run.
    """

    x = np.random.randint(2, size=length)
    fitnessList = [0 for _ in range(iterations+1)]
    fitnessList[0] = get_fitness(x)

    for i in range(1, iterations+1):
        xm = bitflipper(x, pm)
        if get_fitness(xm) > get_fitness(x) or alwaysReplace:
            x = xm
        fitnessList[i] = get_fitness(x)
    return fitnessList


if __name__ == '__main__':
    # 4a
    fitnesses = do_genetic_run(100, 1500, 1/100)
    plt.plot(list(range(0, 1501)), fitnesses)

    #4b
    plt.figure()
    for i in range(10):
        fitnesses = do_genetic_run(100, 1500, 1 / 100)
        plt.plot(list(range(0, 1501)), fitnesses)

    #4c
    fitnesses = do_genetic_run(100, 1500, 1/100, alwaysReplace=True)
    plt.plot(list(range(0, 1501)), fitnesses)
