"""
Authors : Kumarage Tharindu & Fan Lei
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 2
Task : File reader : Provider other package the access to read the data

"""

import numpy as np
import data.data_wrapper as extrct


def main():  # This main() function is used to test the LR Classifier implementation
    data = extrct.read_data()

    model = KMeans(k=3, init_strategy=2, distance_criteria="Euclidean")

    model.cluster(data)

    print(model.centroids)


class KMeans:
    def __init__(self, k=1, init_strategy=1, distance_criteria ="Euclidean"):
        self.k = k
        self.centroid_init_strategy = init_strategy
        self.distance_criteria = distance_criteria
        self.centroids= None
        self.clusters = None

    def logit_sigmoid(self, x, w):
        # Sigmoid function matrix form implementation
        regression_value = np.dot(x, w)

        sigmoid = 1.0 / (1 + np.exp(-regression_value))

        return sigmoid

    def objective_func_val(self, clusters):
        # Calculate sum-of-squared-error for each cluster and get the total sum
        squared_error_list = list()
        for i in range(self.k):
            centroid = clusters[i][0]
            data_points = clusters[i][1]

            # Calculate squared-error between centroid and each data point in the cluster
            point_errors = [self.distance(centroid, x_i) for x_i in data_points]

            # Total error in the cluster
            cluster_error = sum(point_errors)
            squared_error_list.append(cluster_error)

        return sum(squared_error_list)

    def centroid_initialize(self,x):
        centroids = np.empty([self.k, 2], dtype=float)

        #  Initial weights
        if self.centroid_init_strategy == 1:  # random centroid initialization
            index = np.random.choice(x.shape[0], self.k, replace=False)
            centroids = x[index]

        if self.centroid_init_strategy == 2:  # first random and then largest distance initialization
            # random choice for the 1st centroid
            index = np.random.choice(x.shape[0], 1, replace=False)
            centroids[0] = x[index]

            for i in range(1, self.k):
                max_distance = 0
                max_dist_centroid = x[0]
                for x_i in x:
                    if x_i in centroids:
                        continue
                    xi_distance = sum([self.distance(x_i, prev_centroids) for prev_centroids in centroids])

                    if xi_distance > max_distance:
                        max_distance = xi_distance
                        max_dist_centroid = x_i
                centroids[i] = max_dist_centroid

        return centroids

    def distance(self, x, y):  # return the distance between two points
        if self.distance_criteria =="Euclidean":
            return np.linalg.norm(x-y, ord=2)

    def cluster(self, x, display=True):
        #  Train the classifier

        print("k-means clustering on data..")

        print("Initializing centroids..")

        centroids = self.centroid_initialize(x)

        print("Clustering data..")
        clusters = {}
        for i in range(self.k):
            cluster_data = (centroids[i], list())
            clusters[i] = cluster_data

        iteration = 0

        while True:  # Iterate until breaks with terminate convergence

            for x_i in x:  # Assign all the data points to the relevant cluster centroid
                min_distance = self.distance(centroids[0], x_i)
                cluster_no = 0
                cluster_x_i = 0
                for c in centroids:
                    dist = self.distance(c, x_i)
                    if (dist < min_distance):
                        cluster_x_i = cluster_no
                        min_distance = dist
                    cluster_no += 1
                clusters[cluster_x_i][1].append(x_i)

            if display:  # Print the error in each iteration
                print("Iteration: ", iteration, " -- Objective function value: ", self.objective_func_val(clusters))

            # Recompute centroids for ecah cluster
            centroids = list()
            convergent = True

            for i in range(self.k):
                data_points = clusters[i][1]
                data_points.append(clusters[i][0])  # Include centroid
                new_centroid = np.mean(np.array(data_points), axis=0)

                if not ((new_centroid == clusters[i][0]).all()):
                    convergent = False

                centroids.append(new_centroid)
                cluster_data = (new_centroid, list())
                clusters[i] = cluster_data

            iteration += 1

            if convergent:  # Exit iteration if the centroid is not changing
                break

        self.centroids = centroids
        self.clusters = clusters


if __name__ == '__main__':
    main()
