from pdb import set_trace

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def main():
    kmeans_path = 'outputs/k_means_silhouette.csv'
    kmedoids_path = 'outputs/k_medoids_silhouette.csv'

    kmeans = pd.read_csv(kmeans_path, names=['k', 'score']).values[1:]
    plt.figure(1)
    plt.title('K-Means')
    plt.xlim(1, 10)
    plt.xlabel('K')
    plt.ylim(-1, 1)
    plt.ylabel('Silhouette Score')
    plt.plot(kmeans[:, 0] + 1, kmeans[:, 1])
    plt.show()

    kmedoids = pd.read_csv(kmedoids_path, names=['k', 'score']).values[1:]
    plt.figure(2)
    plt.title('K-Medoids')
    plt.xlim(1, 10)
    plt.xlabel('K')
    plt.ylim(-1, 1)
    plt.ylabel('Silhouette Score')
    plt.plot(kmedoids[:, 0] + 1, kmedoids[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
