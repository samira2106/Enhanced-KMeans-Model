# Enhanced-KMeans-Model
This model is based on the research paper 'Improving the Accuracy and Efficiency of the k-means Clustering Algorithm' by K. A. Abdul Nazeer and M. P. Sebastian. The 'Iris' dataset is used for evaluation. 
# Introduction
The implementation of an enhanced version of the k-means clustering algorithm is done based on clustering around the closest pairs of data points in dataset. This enhanced algorithm combines a systematic method for determining initial centroids and an efficient way of assigning data points to clusters, resulting in improved accuracy and efficiency.
# Algorithm Overview
Phase 1: Finding Initial Centroids
1. Set the number of desired clusters, k.
2. Initialize m = 1.
3. Compute the distances between each data point and all other data points.
4. Find the closest pair of data points and form a data-point set Am.
5. Repeat until the number of data points in Am reaches 0.75*n/k.
6. If m < k, find another pair of data points and form another data-point set Am.
7. For each data-point set Am, find the arithmetic mean to obtain the initial centroids.

Phase 2: Assigning Data-Points to Clusters 
1. Compute the distance of each data point to all centroids.
2. For each data point, find the closest centroid and assign it to the corresponding cluster.
3. Set ClusterId and Nearest_Dist for each data point.
4. Recalculate centroids for each cluster.
5. Repeat until convergence criteria are met.
      For each data point:
        Compute its distance from the centroid of the present nearest cluster.
        If the distance is less than or equal to the present nearest distance, the data point stays in the cluster; else, reassign it to the cluster with the nearest centroid.
