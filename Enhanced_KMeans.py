import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits import mplot3d

class KmeansClustering_Enhanced():
    
    def __init__(self, k = 3):
        self.k = k
        self.centroids = None
        self.nearest_dist = None

    def euclidean_distance(data_point1, data_point2):
        return np.sqrt(np.sum((data_point1 - data_point2)**2))


    def fit(self, D, max_iterations = 200):
        Am = []
        length_D = len(D)
        D_copy = D

        for m in range(self.k):
            # closest pair of data points
            distances_points = []

            for i in range(len(D_copy)):
                    for j in range(i, len(D_copy)-1):
                            distances_points.append((i, j+1, KmeansClustering_Enhanced.euclidean_distance(D_copy[i], D_copy[j+1])))

            only_distances = [d[2] for d in distances_points]    
            closest_pair_indices = [(d1, d2) for d1, d2, d3 in distances_points if d3 == np.min(only_distances)][0]

            #adding closest pair to A set and removing it from D set
            A = np.append([D_copy[closest_pair_indices[0]]], [D_copy[closest_pair_indices[1]]], axis=0)
            D_copy = np.delete(D_copy, [closest_pair_indices[0], closest_pair_indices[1]], axis=0)

            #adding the datapoint to A set and deleting it from D set 
            #until length of A reaches 0.75*(n/k)
            mean_A = np.mean(A, axis=0)
            for _ in range(len(D_copy)):
            
                    distance_A_d = []

                    for d in D_copy:
                            distance_A_d.append(KmeansClustering_Enhanced.euclidean_distance(d, mean_A))

                    min_distance_d = D_copy[np.argmin(distance_A_d)]  
                    
                    A = np.append(A, [min_distance_d], axis=0)
                    D_copy = np.delete(D_copy, np.argmin(distance_A_d), axis=0)

                    mean_A = np.mean(A, axis=0)

                    if len(A) >= 0.75*(length_D/self.k):
                            break
            #adding A set to Am set 
            Am.append(A)
            
        #finding initial centroids by averaging data points of A set
        initial_centroids = []
        for m in Am:
                initial_centroids.append(list(np.mean(m, axis=0)))

        #algorithm2
        #distance between each datapoint and initial centroids
        dist_d_centroids = []
        for data_point in D:
                dist_d_c = []
                for centroid in initial_centroids:                
                        dist_d_c.append(KmeansClustering_Enhanced.euclidean_distance(data_point, centroid))

                dist_d_centroids.append(dist_d_c)

        dist_d_centroids = np.array(dist_d_centroids)

        #assigning cluster id with minimum distance and its distance
        cluster_id = np.array([np.argmin(d) for d in dist_d_centroids])
        self.nearest_dist = np.array([dist_d_centroids[i, id] for i, id in enumerate(cluster_id)])

        #recalculating the centroids 
        cluster_indices = []
        for i in range(self.k):
                cluster_indices.append(np.argwhere(cluster_id==i))

        cluster_centroids = []
        for indices in cluster_indices:
                cluster_centroids.append(np.mean(D[indices], axis=0)[0])
        self.centroids = np.array(cluster_centroids)

        for iteration in range(max_iterations):
                #comparing distances between new and previous centroid 
                for i, data_point in enumerate(D):
                        distance = KmeansClustering_Enhanced.euclidean_distance(data_point, self.centroids[cluster_id[i]])

                        if distance <= self.nearest_dist[i]:
                                continue
                        else:
                                dist_d_c = []
                                for centroid in self.centroids:                
                                        dist_d_c.append(KmeansClustering_Enhanced.euclidean_distance(data_point, centroid))

                                cluster_id[i] = np.argmin(dist_d_c)
                                self.nearest_dist[i] = np.min(dist_d_c)

                cluster_centroids_previous = self.centroids
                #recalculating the centroids 
                cluster_indices = []
                for j in range(self.k):
                        cluster_indices.append(np.argwhere(cluster_id==j))

                cluster_centroids = []
                for indices in cluster_indices:
                        cluster_centroids.append(np.mean(D[indices], axis=0)[0])
                self.centroids = np.array(cluster_centroids)
                #checking the convergence requirement
                if abs(np.max(self.centroids - cluster_centroids_previous)) < 0.001:
                        break

        predicted_clusters = np.array(cluster_id)

        return predicted_clusters

#data loading and preprocessing 
data = pd.read_csv('iris.csv')

data['variety'] = data['variety'].replace('Setosa',1)
data['variety'] = data['variety'].replace('Versicolor', 0)
data['variety'] = data['variety'].replace('Virginica', 2)

D = data.to_numpy()[:,:-1]
real_clusters = data.to_numpy()[:, -1]
length_D = len(D)

kmeans = KmeansClustering_Enhanced(k=3)
labels = kmeans.fit(D)

#data visualization
fig = plt.figure(figsize = (8, 5))
ax = plt.axes(projection ="3d")

ax.scatter3D(D[:, 0], D[:, 1], D[:,2], c=labels)
ax.scatter3D(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:,2], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.title('Enhanced KMeans model')
plt.show()

accuracy = (labels==real_clusters).sum()/length_D*100
print(f'The accuracy of Enhanced KMeans model:  {accuracy}%')

#Elbow Method implementation
k_values = range(2, 11)
inertia_values = []

for k in k_values:
    kmeans = KmeansClustering_Enhanced(k=k)
    kmeans.fit(D)
    
    inertia_values.append(np.sum((kmeans.nearest_dist)**2))

# Plot the elbow curve
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

                    
