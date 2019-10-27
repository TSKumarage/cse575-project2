"""
Authors : Kumarage Tharindu & Fan Lei
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 2
Task : File reader : Provider other package the access to read the data

"""

import numpy as np
import data.data_wrapper as extrct
from kmeans_clustering import kmeans
import random
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


def main():
    data = extrct.read_data()

    strategies = [1, 2]
    num_clusters = list(range(2, 11))

    for strategy in strategies:
        for iteration in range(2):
            objective_func_vals = []
            for num_cluster in num_clusters:

                print('-------Strategy {}: k = {}-------'.format(strategy, num_cluster))
                model = kmeans.KMeans(k=num_cluster, init_strategy=strategy, distance_criteria="Euclidean")

                model.cluster(data, display=False)

                objective_func_vals.append(model.objective_func_val(model.clusters))
                print('----------------------------------')
                print()

            plt.title('Strategy {}: Objective function value vs. the number of clusters k'.format(strategy))
            plt.grid(True)
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Objective function value')
            plt.plot(num_clusters, objective_func_vals, marker='o')
            plt.show()


def get_colors(num_colors):

    colors = list(mcolors.CSS4_COLORS.values())
    colors.remove('#000000')

    return [colors[i] for i in random.sample(range(len(colors)), num_colors)]


def plot_cluster(clusters, centroid_list, num_cluster):
    # loop through each cluster

    custom_palette = get_colors(num_colors=num_cluster)

    for i, label in enumerate(clusters.keys()):
        # add data points
        data = clusters[label][1]

        data = np.array(data)

        x = data[:, 0]
        y = data[:, 1]

        plt.scatter(x=x,
                    y=y,
                    color=custom_palette[i],
                    alpha=0.7)

    data = np.array(centroid_list)
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x=x,
                y=y,
                color='#000000',
                alpha=0.9)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
