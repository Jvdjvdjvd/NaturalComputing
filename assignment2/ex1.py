#!/usr/bin/env python3

import random
import math
import numpy as np

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
        np.array([-400, -400]),
        np.array([-410, -410]),
        np.array([-415, -415])
    ]

    velocities = [np.array([-50, -50]) for _ in positions]

    n = len(positions)

    best_positions = positions.copy()
    social_best_position = positions[2]

    r1 = 0.5
    r2 = 0.5
    alpha1 = 1.0
    alpha2 = 1.0
    omega = 2.0

def fitness(p):
    return sum((-x * math.sin(math.sqrt(abs(x)))) for x in positions[p])

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
    py = clamp(positions[p][1] + velocities[p][1], min_pos, max_pos)
    return np.array([px, py])

def run_step():
    for p in range(n):
        velocities[p] = next_velocity(p)
        positions[p] = next_positions(p)

reset()
if __name__ == '__main__':
    print("Answer to A:")
    for i in range(n):
        print("\t{}: fitness {}".format(i + 1, fitness(i)))

    print("Answer to B:")
    print("\tomega = 2")
    reset()
    omega = 2
    run_step()
    for i in range(n):
        print("\t\t{}: fitness {}, position: {}".format(i + 1, fitness(i), positions[i]))
    print("\tomega = 0.5")
    reset()
    omega = 0.5
    run_step()
    for i in range(n):
        print("\t\t{}: fitness {}, position: {}".format(i + 1, fitness(i), positions[i]))

    print("\tomega = 0.1")
    reset()
    omega = 0.1
    run_step()
    for i in range(n):
        print("\t\t{}: fitness {}, position: {}".format(i + 1, fitness(i), positions[i]))
