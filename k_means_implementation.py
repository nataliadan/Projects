# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:43:07 2020

@author: laptop
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class K_Means:
    
    def __init__(self, k, threshold = 0.1):
        '''Initialize K_means class
        Parameters
        ----------
        k : int, number of clusters
        threshold : float, difference in the cluster centers of two consecutive
                    iterations to declare convergence, 0.1 by default
        Attributes
        ----------
        labels : array of int32, labels of each point after clustering
        center_points : array of float, size k X n_features, center points
        iters : int, number of iterations untill convergence
        Returns
        -------
        None.
        '''
        self.k = k
        self.labels = None
        self.center_points = None
        self.iters = 1
        self.threshold = threshold
        
    def fit(self, X):
        '''Function iterates through dataset and cluster points depending on their
           distance from randomly selected k center points.
           Distance calculated by Euclidean distance
        Parameters
        ----------
        X : array of float, given dataset for clustering
        Returns
        -------
        labels : array of int32, contains indices of clusters each sample belongs to
        '''
        # Choose random k center points
        center_row_idx = np.arange(len(X))
        mu = np.random.choice(center_row_idx, self.k, replace = False)
        self.center_points = X[mu]
        old_center_points = np.copy(self.center_points)
        
        k_list = np.arange(self.k)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
        # Create first clusters, update centroids by mean points and get labels
        clusters = self.__create_clusters(X)
        self.center_points = self.__get_centroids(clusters)
        self.labels = self.__get_cluster_labels(clusters)
        self.__plot_results(clusters)

        # Repeate until convergence
        while not self.__is_converged(old_center_points):
            old_center_points = self.center_points
            clusters = self.__create_clusters(X)
            self.center_points = self.__get_centroids(clusters)
            self.labels = self.__get_cluster_labels(clusters)
            self.iters += 1
            self.__plot_results(clusters)

        return self.labels

    def __create_clusters(self, X):
        '''Assign closest samples to given centroids by indices
        Parameters
        ----------
        X : array of float, given dataset for clustering
        Returns
        -------
        clusters : dictionary, each key is a cluster and each value is list of samples's
                   indices accordingly to closest distance to relevant centroid
        '''
        clusters = {key: [] for key in range(self.k)}
        for idx, sample in enumerate(X):
            centroid_idx = self.__closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def __closest_centroid(self, p1):
        '''Find to wich centroid, given point has closest euclidean distance 
        Parameters
        ----------
        p1 : array of float, one sample from dataset
        Returns
        -------
        closest_index : int, row index of the closest centroid to given point p1
        '''
        distances = [euclidean_distance(p1, point) for point in self.center_points]
        closest_index = np.argmin(distances)
        return closest_index

    def __get_centroids(self, clusters):
        '''Calculates an average point for each cluster to update the center points
        Parameters
        ----------
        clusters : dictionary, each key is a cluster and each value is list of samples's
                   indices accordingly to closest distance to relevant centroid
        Returns
        -------
        centroids : array of float, size k X n_features, mean point of each cluster
        '''
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in clusters.items():
            cluster_mean = np.mean(X[cluster], axis=0)  # Average along columns: average point
            centroids[cluster_idx] = cluster_mean
        return centroids

    def __get_cluster_labels(self, clusters):
        '''Update label values accordingly to present clusters
        Parameters
        ----------
        clusters : dictionary, each key is a cluster and each value is list of samples's
                   indices accordingly to closest distance to relevant centroid
        Returns
        -------
        labels : array of float, cluster value for each sample
        '''
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in clusters.items():
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def __is_converged(self, old_center_points):
        '''Provides stopping condition for iterations by calculating the distances
           between previous and present centroids(self.center_points)
        Parameters
        ----------
        old_center_points : array of float, center points from previous iteration
        Returns
        -------
        boolean : True if previous and present centroids are the same points,
                  False otherwise
        '''
        distances = [euclidean_distance(old_center_points[i], self.center_points[i]) for i in range(self.k)]
        return sum(distances) < self.threshold

    def __plot_results(self, clusters):
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown',
                  'orange', 'purple', 'gray']
        assert len(colors) > self.k, f'Maximum number of clusters for plot can be {len(colors)}'
        i = 0
        for cluster_idx, cluster in clusters.items():
            points = X[cluster]
            plt.scatter(points[:,0], points[:,1], color = colors[i])
            plt.scatter(self.center_points[:,0], self.center_points[:,1], s=100, color="black", marker = 'x'); # Show the centres
            i += 1
        plt.title(f"My K_means {self.iters} iteration")
        plt.show()
        
def optimal_k_value(X): 
    '''Implementing elbow method to determine the optimal value of K using Sklearn KMeans
    Parameters
    ----------
    X : dataset
    Returns
    -------
    None. PLots K value vs. Squared error
    '''
    cost =[] 
    for i in range(1, 11): 
        KM = KMeans(n_clusters = i, max_iter = 500) 
        KM.fit(X) 
        # calculates squared error for the clustered points 
        # KM.inertia_ : Sum of squared distances of samples to their closest cluster center.
        cost.append(KM.inertia_)        
    # plot the cost against K values 
    plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
    plt.xlabel("Value of K") 
    plt.ylabel("Sqaured Error (Cost)") 
    plt.show()     

if __name__ == "__main__":

    iris = load_iris()
    X = iris.data
    np.random.seed(42)
    km = K_Means(k = 3)
    target = km.fit(X)
    plt.scatter(X[:,0], X[:,1], c = iris.target, cmap='Paired')
    plt.title("Iris dataset")
    plt.show()  

    # KMeans from Sklearn
    kmeans = KMeans(n_clusters = 3).fit(X)
    plt.scatter(X[:,0], X[:,1], c = kmeans.labels_, cmap='Paired')
    plt.title("Sklearn KMeans dataset")
    plt.show() 
    print(f"Number of iterations until convergence: mine {km.iters}, sklearn {kmeans.n_iter_}")
    optimal_k_value(X)
