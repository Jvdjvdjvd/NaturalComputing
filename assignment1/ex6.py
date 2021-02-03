#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import operator

cities = [
    (0.2554,    18.2366),
    (0.4339,    15.2476),
    (0.7377,    8.3137),
    (1.1354,    16.5638),
    (1.5820,    17.3030),
    (2.0913,    9.2924),
    (2.2631,    17.3392),
    (2.6373,    2.6425),
    (3.0040,    19.5712),
    (3.6684,    14.8018),
    (3.8630,    13.7008),
    (4.2065,    9.8224),
    (4.8353,    2.0944),
    (4.9785,    3.1596),
    (5.3754,    17.6381),
    (5.9425,    6.0360),
    (6.1451,    3.8132),
    (6.7782,    11.0125),
    (6.9223,    7.7819),
    (7.5691,    0.9378),
    (7.8190,    13.1697),
    (8.3332,    5.9161),
    (8.5872,    7.8303),
    (9.1224,    14.5889),
    (9.4076,    9.7166),
    (9.7208,    8.1154),
    (10.1662,   19.1705),
    (10.7387,   2.0090),
    (10.9354,   5.1813),
    (11.3707,   7.2406),
    (11.7418,   13.6874),
    (12.0526,   4.7186),
    (12.6385,   12.1000),
    (13.0950,   13.6956),
    (13.3533,   17.3524),
    (13.8794,   3.9479),
    (14.2674,   15.8651),
    (14.5520,   17.2489),
    (14.9737,   13.2245),
    (15.2841,   1.4455),
    (15.5761,   12.1270),
    (16.1313,   14.2029),
    (16.4388,   16.0084),
    (16.7821,   9.4334),
    (17.3928,   12.9692),
    (17.5139,   6.4828),
    (17.9487,   7.5563),
    (18.3958,   19.5112),
    (18.9696,   19.3565),
    (19.0928,   16.5453)
]


def gen_candidate():
    return np.random.permutation(range(0, len(cities)))

def gen_two_assending_indices():
    length = len(cities)
    
    idx1 = random.randrange(0, length)
    idx2 = random.randrange(0, length)
    while idx1 == idx2:
        idx2 = random.randrange(0, length)
    
    if (idx1 > idx2):
        tmp = idx1
        idx1 = idx2
        idx2 = tmp
    
    return (idx1, idx2)

def fitness_ga(candidate):
    currentCity = candidate[0]
    distance = 0
    for nextCity in candidate[1:]:
        distance += math.dist(cities[currentCity], cities[nextCity])
        currentCity = nextCity

    return distance

def crossover_ga(parent1, parent2):
    length = len(parent1)

    (cut1, cut2) = gen_two_assending_indices()

    offspring1 = [None] * length
    offspring2 = [None] * length
    offspring1[cut1:cut2] = parent1[cut1:cut2]
    offspring2[cut1:cut2] = parent2[cut1:cut2]
    
    j = cut2
    q = cut2
    for i in range(cut2, length + cut2):
        i2 = i % length
        j2 = j % length
        q2 = q % length
        
        if parent2[i2] not in offspring1:
            offspring1[j2] = parent2[i2]
            j += 1

        if parent1[i2] not in offspring2:
            offspring2[q2] = parent1[i2]
            q += 1
    
    return (offspring1, offspring2)
    
def mutation_ga(candidate):
    (idx1, idx2) = gen_two_assending_indices()
    tmp = candidate[idx1]
    candidate[idx1] = candidate[idx2]
    candidate[idx2] = tmp


def run_ga(iterations, n, k, memetic=False):
    candidates = [gen_candidate() for _ in range(n)]
    candidates = [(fitness_ga(c), c) for c in candidates]

    current_best = candidates[0][0]
    for _ in range(iterations):
        participants = random.choices(candidates, k = k)
        participants.sort(key=operator.itemgetter(0))
        [parent1, parent2] = participants[:2]
        (offspring1, offspring2) = crossover_ga(parent1[1], parent2[1])

        mutation_ga(offspring1)
        mutation_ga(offspring2)

        if memetic:
            pass

        candidates.append((fitness_ga(offspring1), offspring1))
        candidates.append((fitness_ga(offspring2), offspring2))

        candidates.sort(key=operator.itemgetter(0))
        candidates = candidates[:n]
        
        if (current_best > candidates[0][0]):
            current_best = candidates[0][0]
            print(candidates[0][0])
    
    return candidates[0]
        
def swap_positions(candidate, i, j):
    new_candidate = candidate.copy()
    new_candidate[i], new_candidate[j] = new_candidate[j], new_candidate[i]
    return new_candidate
    
def get_single_swaps(candidate):
    possible_candidates = [candidate]
    for i in range(len(candidate)-1):
        for j in range(i+1, len(candidate)):
            possible_candidates.append([swap_positions(candidate, i,j)])
    return possible_candidates

def do_local_search(candidate):
    neighbourhood = get_single_swaps(candidate)
    fitnesses = [fitness_ga(c) for c in neighbourhood]
    best = min(fitnesses)
    best_idx = neighbourhood.index(best)
    best_of_all = neighbourhood[best_idx]
    return best_of_all



if __name__ == '__main__':
    print(run_ga(50000, 50, 20))
