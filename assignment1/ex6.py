#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import operator


def read_cities(fileLocation, rownr = True):
    """
    reads citiy coordinates from a file. MAY NOT CONTAIN HEADER!
    """
    with open(fileLocation) as f:
        cities = f.readlines()
        cities = [c.strip().split(' ') for c in cities]
        if rownr:
            firstIDX = 1
        else:
            firstIDX = 0
        cities = [(float(c[firstIDX]), float(c[-1])) for c in cities]
    return cities

cities = read_cities("assignment1/file-tsp.txt", False) #for given data
cities = read_cities("assignment1/att48.tsp") #for other data (http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/att48.tsp) - without header and footer


def gen_candidate():
    """
    Generate a random iteration of cities
    """
    return np.random.permutation(range(0, len(cities)))

def gen_two_assending_indices():
    """
    returns two random indices where the first one is smaller than the seqond one. The indices are between 0 and len(cities).
    """

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
    """
    Returns the fitness of a candidate by calculating the distance when you walk from city to city
    """
    currentCity = candidate[0]
    distance = 0
    for nextCity in candidate[1:]:
        distance += math.dist(cities[currentCity], cities[nextCity])
        currentCity = nextCity

    return distance

def crossover_ga(parent1, parent2):
    """
    Does crossover between two partents. A slice is selected from both parents and the missing numbers are completed based on the other parent.
    """
    length = len(parent1)

    (cut1, cut2) = gen_two_assending_indices() #get two cuts

    offspring1 = [None] * length
    offspring2 = [None] * length
    offspring1[cut1:cut2] = parent1[cut1:cut2] #get the initial offspring by taking a parent slice
    offspring2[cut1:cut2] = parent2[cut1:cut2]
    
    j = cut2
    q = cut2
    # fill the offspring based on their parents
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
    """
    performs pointwise mutation on a candidate
    """
    (idx1, idx2) = gen_two_assending_indices()
    candidate[idx1], candidate[idx2]  = candidate[idx2], candidate[idx1]

       
def swap_positions(candidate, i, j):
    """
    swaps the positions of a candidate
    """
    new_candidate = candidate.copy()
    new_candidate[i], new_candidate[j] = new_candidate[j], new_candidate[i]
    return new_candidate
    
def get_single_swaps(candidate):
    """
    gives all possible candidates where two positions are swapped.
    """
    possible_candidates = [candidate]
    for i in range(len(candidate)-1):
        for j in range(i+1, len(candidate)):
            possible_candidates.append(swap_positions(candidate, i,j))
    return possible_candidates

def do_local_search(candidate):
    """
    performs a local search by generating all neighbours of the candidate and selecting the best one
    """
    neighbourhood = get_single_swaps(candidate)
    fitnesses = [fitness_ga(c) for c in neighbourhood]
    best = min(fitnesses)
    best_idx = fitnesses.index(best)
    best_of_all = neighbourhood[best_idx]
    return best_of_all

def run_ga(iterations, n, k, memetic=False):
    """
    runs the genetic algorithm
    @param n: the initial population size
    @param k: amount of random selected participants to look in for offspring generation
    """

    candidates = [gen_candidate() for _ in range(n)] #get initial candidates
    if memetic:
        candidates = [do_local_search(c) for c in candidates] # do a local search if it is memetic
    candidates = [(fitness_ga(c), c) for c in candidates]

    current_best = candidates[0][0]
    for _ in range(iterations):
        participants = random.choices(candidates, k = k) #select k random participants
        participants.sort(key=operator.itemgetter(0))
        [parent1, parent2] = participants[:2]
        (offspring1, offspring2) = crossover_ga(parent1[1], parent2[1]) #generate offspring by crossover

        mutation_ga(offspring1) #mutate offspring
        mutation_ga(offspring2)

        if memetic:
            offspring1 = do_local_search(offspring1) #do a local search on the offspring
            offspring2 = do_local_search(offspring2)

            

        candidates.append((fitness_ga(offspring1), offspring1)) #add offspring to canidates
        candidates.append((fitness_ga(offspring2), offspring2))

        candidates.sort(key=operator.itemgetter(0)) 
        candidates = candidates[:n] #select the best candidates as new population
        
        if (current_best > candidates[0][0]):
            current_best = candidates[0][0]
            print(candidates[0][0])
    
    return candidates[0]

if __name__ == '__main__':
    print(cities)
    print(run_ga(50000, 50, 20))
