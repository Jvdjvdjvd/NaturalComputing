#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from random import uniform

def get_quantization_error(datapoints, centroids, points):
    err = 0
    for i in range(len(centroids)):
        c = centroids[i]
        pts = [datapoints[idx] for idx in points[i]]
        err = err + sum([np.linalg.norm(p - c) / len(pts) for p in pts])

    return err / len(centroids) 

class kmeans:
    def __init__(self, groups):
        self.groups=groups
        self.centroids = None

    def train_kmeans(self, trainData, iterations = 3) -> object:
        """
        Trains the centroids of the kmeans class using the given trainData (shape = (S,F), samples, features)
        todo: in this version it trains for a set amount of iterations. In future versions it would be good to train untill convergence
        :param trainData: data to find centers with
        :param iterations: How many iterations to train on
        :return: log list of all the centroids per iteration.
        """
        min_values = np.min(trainData, axis=0) #get range to place centroids in
        max_values = np.max(trainData, axis=0)

        centroid_log = []

        self.centroids = self.generate_centroids(min_values, max_values) #generate inital centroids

        for it in range(iterations):
            distances = self.get_distances_to_centroids(trainData)
            closest_centroids = np.argmin(distances, axis=1)  # find nearest centroid per sample

            centroid_center = []
            centroid_points = []
            for i in range(self.groups): # update the centroids
                idxes = [j for j in range(len(trainData)) if closest_centroids[j] == i]

                if len(idxes) > 0: # do not change if no samples are present this run
                    center = np.mean(trainData[idxes], axis=0)
                    self.centroids[i] = center

                centroid_center.append(self.centroids[i])
                centroid_points.append(idxes)

            centroid_log.append((centroid_center, centroid_points))
        return centroid_log

    def get_distances_to_centroids(self, data):
        distances = np.zeros((len(data), self.groups))
        for i, place in enumerate(data):
            distances[i] = [np.linalg.norm(self.centroids[j] - place) for j in range(self.groups)]
        return distances

    def generate_centroids(self, max_values, min_values):
        difference = max_values - min_values
        centroids = np.array(
            [np.ones_like(min_values) * min_values + np.random.random(len(min_values)) * difference for _ in
             range(self.groups)])
        return centroids

class pso:
    def __init__(self, groups, alpha1, alpha2, omega):
        self.groups = groups
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.omega = omega

    def generate_centroids(self):
        difference = self.max_values - self.min_values
        centroids = np.array(
            [np.ones_like(self.min_values) * self.min_values + np.random.random(len(self.min_values)) * difference for _ in
             range(self.groups)])
        return centroids

    def generate_velocities(self):
        difference = self.max_values - self.min_values

        velocities = np.array(
            [(np.random.random(len(self.min_values)) - 0.5) * (difference / 3) for _ in
             range(self.groups)])

        return velocities

    def get_centroid_assignment(self, data, centroids):
        distances = np.zeros((len(data), self.groups))
        for i, place in enumerate(data):
            distances[i] = [np.linalg.norm(c - place) for c in centroids]

        min_distances = np.argmin(distances, axis=1)

        assignment = [None for _ in range(len(centroids))]
        for i in range(self.groups):
            assignment[i] = np.where(min_distances == i)[0]

        return np.asarray(assignment, dtype="object")


    def train_pso(self, trainData, n = 5, iterations = 3):
        # Initialize values before running iterations
        self.min_values = np.min(trainData, axis=0)
        self.max_values = np.max(trainData, axis=0)

        self.centroids = np.asarray([
            self.generate_centroids()
            for _ in range(n)
        ])

        self.velocities = np.asarray([
            self.generate_velocities()
            for _ in range(n)
        ])

        self.centroid_assignments = [
            self.get_centroid_assignment(trainData, c)
            for c in self.centroids
        ]

        self.fitnesses = np.asarray([self.fitness(trainData, i) for i in range(n)])
        self.best_fitnesses = self.fitnesses.copy()
        self.global_best_fitness = np.argmin(self.best_fitnesses)

        self.best_centroids = self.centroids.copy()
        self.best_assignments = self.centroid_assignments.copy()

        # run iterations
        for i in range(iterations):
            self.run_step(trainData, n)
        
        return (self.best_centroids[self.global_best_fitness], self.best_assignments[self.global_best_fitness])

    def fitness(self, trainData, i):
        return get_quantization_error(trainData, self.centroids[i], self.centroid_assignments[i])

    def next_velocity(self, p):
        r1 = uniform(0, 1)
        r2 = uniform(0, 1)

        inertia = self.omega * self.velocities[p]
        personal_influence = self.alpha1 * r1 * (self.best_centroids[p] - self.centroids[p])
        social_influence =  self.alpha2 * r2 * (self.best_centroids[self.global_best_fitness] - self.centroids[p])

        return inertia + personal_influence + social_influence

    def next_centroids(self, p):
        return np.clip(self.centroids[p] + self.velocities[p], self.min_values, self.max_values)

    def run_step(self, trainData, n):
        for i in range(n):
            self.velocities[i] = self.next_velocity(i)
            self.centroids[i] = self.next_centroids(i)

            self.centroid_assignments[i] = self.get_centroid_assignment(trainData, self.centroids[i])

            self.fitnesses[i] = self.fitness(trainData, i)
            
            if self.fitnesses[i] < self.best_fitnesses[self.global_best_fitness]:
                self.global_best_fitness = i

            if self.fitnesses[i] < self.best_fitnesses[i]:
                self.best_centroids[i] = self.centroids[i]
                self.best_assignments[i] = self.centroid_assignments[i]
                self.best_fitnesses[i] = self.fitnesses[i]

def plot(name, dataset, centroids, points, writeToFile = True):
    # Plot colored scatterplot of every centroid with data points 
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    clrs = ["r", "g", "b", "c", "m", "y"]

    for i in range(len(centroids)):
        c = centroids[i]
        pts = points[i]
        if len(pts) == 0:
            pts_x = []
            pts_y = []
        else:
            (pts_x, pts_y) = zip(*[dataset[idx] for idx in pts])
        plt.scatter(pts_x, pts_y, marker="o", edgecolor='k', c=clrs[i], label = f"Cluster {i} data points")
        plt.scatter(c[0], c[1], marker="v", edgecolor='k', c=clrs[i], label = f"Centroid {i}")

    err = get_quantization_error(dataset, centroids, points)

    print(f"{name} quantization error: {err}")

    plt.legend()
    plt.annotate(f"Quantization error: {err}",
            xy=(0, 3), xycoords='figure pixels')

    if writeToFile:
        plt.savefig(f"{name}.png")
    else:
        plt.show()

def run_kmeans(datasetName, groups, dataset, iterationSet = [1, 2, 10, 25, 60], average = 30):
    errors = []
    for i in iterationSet:
        err = 0
        for a in range(average):
            bla = kmeans(groups)
            lg = bla.train_kmeans(dataset, iterations=i)
            (centroids, points) = lg[-1]

            err = err + get_quantization_error(dataset, centroids, points)

        err = err / float(average)

        print(f"{datasetName} kmeans quantization error, {i} iterations, averaged over {average} runs: {err}")


def run_pso(datasetName, groups, dataset, iterationSet = [1, 2, 10, 25, 60], average = 30):
    for i in iterationSet:
        err = 0
        for a in range(average):
            # use parameters from paper
            bla = pso(groups, 1.49, 1.49, 0.72)
            (centroids, points) = bla.train_pso(dataset, n = 15, iterations = i)

            err = err + get_quantization_error(dataset, centroids, points)

        err = err / float(average)

        print(f"{datasetName} PSO quantization error, {i} iterations, averaged over {average} runs: {err}")

        # plot(f"pso_{i}", dataset, centroids, points)

def generate_artifical_set1():
    return np.asarray([np.asarray([uniform(-1, 1), uniform(-1, 1)]) for _ in range(400)])

if __name__ == '__main__':
    iris = datasets.load_iris()['data']
    artificial1 = generate_artifical_set1()

    run_kmeans("artifical_set1", 2, artificial1)
    run_pso("artifical_set1", 2, artificial1)

    run_kmeans("iris", 3, iris)
    run_pso("iris", 3, iris)
