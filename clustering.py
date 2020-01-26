import random
from math import sqrt
import threading

from queue import Queue
from multiprocessing import cpu_count

import numpy as np
from timeit import default_timer as timer


def lb_keogh(series_a, series_b, window_size):
    """Lower-bounding algorithm for DTW. For details please go here:
    https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
    
    Parameters
    ----------
    series_a : array_like
        The time series to compute the lower bound for.
    series_b : array_like
        The time series to compute the lower bound for.
    window_size : int
        The window size for DTW computation.
    
    Returns
    -------
    float :
        The lower bound.
    """
    lb_sum = 0
    for index, value in enumerate(series_a):

        # figure out windowing
        start_index = 0
        if index - window_size >= 0:
            start_index = index - window_size

        stop_index = index + window_size

        lower_bound = min(series_b[start_index:stop_index])
        upper_bound = max(series_b[start_index:stop_index])

        if value > upper_bound:
            lb_sum = lb_sum + (value - upper_bound) ** 2
        elif value < lower_bound:
            lb_sum = lb_sum + (value - lower_bound) ** 2

    return sqrt(lb_sum)


def dtw_distance(series_a, series_b, window_size):
    """Computes the DTW distance between two time series given a window
    size.
    
    Parameters
    ----------
    series_a : array_like
        The time series to compute the lower bound for.
    series_b : array_like
        The time series to compute the lower bound for.
    window_size : int
        The window size for DTW computation.
    
    Returns
    -------
    float :
        The DTW distance.
    """
    dtw = {}
    difference = abs(len(series_a) - len(series_b))
    w = max(window_size, difference)

    for i in range(-1, len(series_a)):
        for j in range(-1, len(series_b)):
            dtw[(i, j)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(series_a)):
        for j in range(max(0, i - w), min(len(series_b), i + w)):
            dist = (series_a[i] - series_b[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i-1, j)],dtw[(i, j-1)], dtw[(i-1, j-1)])

    return sqrt(dtw[len(series_a) - 1, len(series_b) - 1])


class DTWClustering(object):
    """Clusters a list of time series to form the desired number of clusters
    using DTW with LB_Keogh.
    
    Attributes
    ----------
    data : list
        The list of time series to cluster.
    k : int
        The desired number of clusters to compute.
    window_size : int
        The window size used to compute distances.
    max_iters : int, Default 10
        The maximum number of iterations to refine the clusters in.
    n_jobs : int, Default All cores
        The number of cpu cores to use.
    verbose : bool, Default True
        Flag to indicate if runtime output should be shown.
    centroids : array_like
        The time series centroids with index specifying the cluster group.
    clusters : dict
        The cluster assignments for the series. Key is the centroid index.
    """
    def __init__(self, data, k, max_iters=10, window_size=4,
        n_jobs=cpu_count(), verbose=True):
        self.data = data
        self.k = k
        self.n_jobs = n_jobs
        self.max_iters = max_iters
        self.window_size = window_size
        self.threads = []
        self.verbose = verbose
        self.queue = Queue()
        self.clusters = {}
        self.centroids = []
    
    def __compute_distance(self, series_index):
        """Computes the DTW distance for a given time series index."""
        series = self.data[series_index]
        minimum_distance = float('inf')
        closest_cluster = None
        for cluster_index, cluster_series in enumerate(self.centroids):
            if lb_keogh(series, cluster_series, self.window_size):
                current_distance = dtw_distance(series, cluster_series,
                    self.window_size)
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    closest_cluster = cluster_index

        if closest_cluster not in self.clusters:
            self.clusters[closest_cluster] = []

        self.clusters[closest_cluster].append(series_index)
    
    def __dequeue_worker(self):
        """Worker function for parallelism."""
        while True:
            item = self.queue.get()
            if item is None:
                break
            
            self.__compute_distance(item)
            self.queue.task_done()
        
    def __init_workers(self):
        """Create workers based on n_jobs."""
        for i in range(self.n_jobs):
            thread = threading.Thread(target=self.__dequeue_worker)
            thread.start()
            self.threads.append(thread)
            
    def __stop_workers(self):
        """Stops the worker threads."""
        for i in range(self.n_jobs):
            self.queue.put(None)

        for thread in self.threads:
            thread.join()
        
    def train(self):
        """Clusters the time series together."""
        self.centroids = random.sample(list(self.data), self.k)        
        self.__init_workers()
        
        for iteration in range(self.max_iters):
            start = timer()
            self.clusters = {}
            
            for series_index, series in enumerate(self.data):                
                self.queue.put(series_index, False)
            
            if self.verbose: 
                print(timer() - start, 'queue placement complete')
            
            self.queue.join()
            
            if self.verbose:
                print(timer() - start, 'computations complete')
                
            #recalculate centroids of clusters
            for key in self.clusters:
                cluster_sum = 0
                for k in self.clusters[key]:
                    cluster_sum = cluster_sum + self.data[k]
                self.centroids[key] = [
                    m / len(self.clusters[key]) for m in cluster_sum
                ]
                
            if self.verbose:
                print(timer() - start, 'iteration complete')
        
        self.queue.join()
        self.__stop_workers()

    def predict(self, series):
        """Predicts the cluster that the provided time series belongs to.

        Parameters
        ----------
        series : array_like
            The series to predict.

        Returns
        -------
        (float, int) :
            The distance and cluster index.
        """
        """Computes the DTW distance for a given time series index."""
        minimum_distance = float('inf')
        closest_cluster = None
        for cluster_index, cluster_series in enumerate(self.centroids):
            if lb_keogh(series, cluster_series, self.window_size):
                current_distance = dtw_distance(series, cluster_series, self.window_size)
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    closest_cluster = cluster_index

        return (minimum_distance, cluster_index)
