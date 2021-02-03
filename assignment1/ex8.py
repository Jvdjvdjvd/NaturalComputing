#!/usr/bin/env python3

import random
import math
import operator

import numpy as np
import matplotlib.pyplot as plt
import deap as dp
import deap.algorithms as algorithms
import deap.gp as gp
import deap.creator as creator
import deap.base as base
import deap.tools as tools
import matplotlib.pyplot as plt


data = {
        -1.0: 0.0,
        -0.9: -0.1629,
        -0.8: -0.2624,
        -0.7: -0.3129,
        -0.6: -0.3264,
        -0.5: -0.3125,
        -0.4: -0.2784,
        -0.3: -0.2289,
        -0.2: -0.1664,
        -0.1: -0.0909,
        0: 0.0,
        0.1: 0.1111,
        0.2: 0.2496,
        0.3: 0.4251,
        0.4: 0.6496,
        0.5: 0.9375,
        0.6: 1.3056,
        0.7: 1.7731,
        0.8: 2.3616,
        0.9: 3.0951,
        1.0: 4.0
}


def fitness(individual, points):
    fn = toolbox.compile(expr=individual)
    # Compute the aboslute sum of errors
    return (math.fsum([abs(data[x] - fn(x)) for x in points]), )


def zerodiv(left, right):
    if right == 0:
        return 0
    else:
        return left / right

def zerolog(x):
    if x <= 0:
        return 0
    else:
        return math.log(x)

if __name__ == '__main__':
    # This implementation is based on the symbolic regression introduction from
    # DEAP at https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

    # The symbolic expression has a single input argument.
    pset = gp.PrimitiveSet("ex8", 1)

    # Define the function and terminal set.
    pset.renameArguments(ARG0="x")
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(zerolog, 1)
    pset.addPrimitive(math.exp, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(zerodiv, 2)

    # We want to find an expression that minimizes the absolute error hence -1.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # Use a tree structure.
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Configure evolution process parameters.
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness, points=[x/10. for x in range(-10,11)])
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)

    # Statistics.
    stats_fit = dp.tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Run the evolution.
    cxpb = 0.7
    mutpb = 0.0
    ngen = 50
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)

    # Plot the results.
    gen = log.select("gen")
    fit_mins = log.chapters["fitness"].select("min")
    size_avgs = log.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
