import numpy as np


class kmeans:
    def __init__(self, groups):
        self.groups=groups
        self.centroids = None

    def train_kmeans(self, trainData, iterations = 20) -> object:
        """
        Trains the centroids of the kmeans class using the given trainData (shape = (S,F), samples, features)
        todo: in this version it trains for a set amount of iterations. In future versions it would be good to train untill convergence
        :param trainData: data to find centers with
        :param iterations: How many iterations to train on
        :return: log list of all the centroids per iteration.
        """
        min_values = np.min(trainData, axis=0) #get range to place centroids in
        max_values = np.max(trainData, axis=0)
        centroid_log = np.zeros((iterations, self.groups, len(min_values))) #create a log file

        self.centroids = self.generate_centroids(min_values, max_values) #generate inital centroids

        for it in range(iterations):
            centroid_log[it] = self.centroids.copy()
            for i in range(self.groups): # update the centroids
                distances = self.get_distances_to_centroids(trainData)
                closest_centroids = np.argmin(distances, axis=1)  # find nearest centroid per sample
                idxes = [j for j in range(len(trainData)) if closest_centroids[j] == i]
                if len(idxes) > 0: # do not change if no samples are present this run
                    center = np.mean(trainData[idxes], axis=0)
                    self.centroids[i] = center
        return centroid_log

    def get_distances_to_centroids(self, data):
        distances = np.zeros((len(data), self.groups))
        for i, place in enumerate(data):
            distances[i] = [np.linalg.norm(self.centroids[j] - place) for j in range(self.groups)]
        return distances

    def generate_centroids(self, min_values, max_values):
        """
        amount = amount of centroids
        min_values: minimal value for each dimension
        max_values: maximal value for each dimenstion
        """
        difference = max_values - min_values
        centroids = np.array(
            [np.ones_like(min_values) * min_values + np.random.random(len(min_values)) * difference for _ in
             range(self.groups)])
        return centroids

if __name__ == '__main__':
    data = np.random.random((500, 2))
    bla = kmeans(5)
    lg = bla.train_kmeans(data)






