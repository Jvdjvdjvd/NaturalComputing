#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def reset():
    global min_pos
    global max_pos
    global positions
    global velocities
    global n
    global best_positions
    global social_best_position
    global r1
    global r2
    global alpha1
    global alpha2
    global omega

    min_pos = -500
    max_pos = 500

    positions = [
        np.array([20])
    ]

    velocities = [np.array([10]) for _ in positions]

    n = len(positions)

    best_positions = positions.copy()
    social_best_position = positions[0]

    r1 = 0.5
    r2 = 0.5
    alpha1 = 1.0
    alpha2 = 1.0
    omega = 2.0

def fitness(p):
    return p[0] * p[0]

def clamp(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val

def next_velocity(p):
    inertia = omega * velocities[p]
    personal_influence = alpha1 * r1 * (best_positions[p] - positions[p])
    social_influence =  alpha2 * r2 * (social_best_position - positions[p])
    return inertia + personal_influence + social_influence

def next_positions(p):
    px = clamp(positions[p][0] + velocities[p][0], min_pos, max_pos)
    return np.array([px])

def run_step():
    global social_best_position

    for p in range(n):
        velocities[p] = next_velocity(p)
        positions[p] = next_positions(p)

        social_best_fitness = fitness(social_best_position)
        best_fitness = fitness(best_positions[p])
        new_fitness = fitness(positions[p])

        social_best_position = social_best_position if new_fitness > social_best_fitness else positions[p]
        best_positions[p] = best_positions[p] if new_fitness > best_fitness else positions[p]

reset()
if __name__ == '__main__':
    iterations = 25
    results = {
        0.1: [],
        0.25: [],
        0.5: [],
        0.75: [],
        0.99: [],
    }

    for k in results:
        reset()
        for _ in range(iterations):
            omega = k
            results[k].append(fitness(positions[0]))
            run_step()

    fig = plt.figure()
    plt.plot(results[0.1], 'r', label="Omega = 0.1")
    plt.plot(results[0.25], 'g', label="Omega = 0.25")
    plt.plot(results[0.5], 'b', label="Omega = 0.5")
    plt.plot(results[0.75], 'm', label="Omega = 0.75")
    plt.plot(results[0.99], 'k', label="Omega = 0.99")
    leg = plt.legend(loc='best', ncol=3, mode="expand", shadow=True, fancybox=True)
    fig.suptitle('Single particle swarm, omega < 1', fontsize=20)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Fitness', fontsize=16)
    fig.savefig('single-particle-swarm.png')
