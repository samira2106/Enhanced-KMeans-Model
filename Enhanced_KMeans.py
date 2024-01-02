import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d

def euclidean_distance(data_point1, data_point2):
        return np.sqrt(np.sum((data_point1 - data_point2)**2))

#data loading and preprocessing 
data = pd.read_csv('iris.csv')

data['variety'] = data['variety'].replace('Setosa',1)
data['variety'] = data['variety'].replace('Versicolor', 0)
data['variety'] = data['variety'].replace('Virginica', 2)

D = data.to_numpy()[:,:-1]

D_copy = D
k = 3
length_D = len(D)
Am = []
max_iterations = 200

for m in range(k):
        # closest pair of data points
        distances_points = []

        for i in range(len(D_copy)):
                for j in range(i, len(D_copy)-1):
                        distances_points.append((i, j+1, euclidean_distance(D_copy[i], D_copy[j+1])))

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
                        distance_A_d.append(euclidean_distance(d, mean_A))

                min_distance_d = D_copy[np.argmin(distance_A_d)]  
                
                A = np.append(A, [min_distance_d], axis=0)
                D_copy = np.delete(D_copy, np.argmin(distance_A_d), axis=0)

                mean_A = np.mean(A, axis=0)

                if len(A) >= 0.75*(length_D/k):
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
                dist_d_c.append(euclidean_distance(data_point, centroid))

        dist_d_centroids.append(dist_d_c)

dist_d_centroids = np.array(dist_d_centroids)

#assigning cluster id with minimum distance and its distance
cluster_id = np.array([np.argmin(d) for d in dist_d_centroids])
nearest_dist = np.array([dist_d_centroids[i, id] for i, id in enumerate(cluster_id)])

#recalculating the centroids 
cluster_indices = []
for i in range(k):
        cluster_indices.append(np.argwhere(cluster_id==i))

cluster_centroids = []
for indices in cluster_indices:
        cluster_centroids.append(np.mean(D[indices], axis=0)[0])
cluster_centroids = np.array(cluster_centroids)

for iteration in range(max_iterations):
        #comparing distances between new and previous centroid 
        for i, data_point in enumerate(D):
                distance = euclidean_distance(data_point, cluster_centroids[cluster_id[i]])

                if distance <= nearest_dist[i]:
                        continue
                else:
                        dist_d_c = []
                        for centroid in cluster_centroids:                
                                dist_d_c.append(euclidean_distance(data_point, centroid))

                        cluster_id[i] = np.argmin(dist_d_c)
                        nearest_dist[i] = np.min(dist_d_c)

        cluster_centroids_previous = cluster_centroids
        #recalculating the centroids 
        cluster_indices = []
        for j in range(k):
                cluster_indices.append(np.argwhere(cluster_id==j))

        cluster_centroids = []
        for indices in cluster_indices:
                cluster_centroids.append(np.mean(D[indices], axis=0)[0])
        cluster_centroids = np.array(cluster_centroids)
        #checking the convergence requirement
        if abs(np.max(cluster_centroids - cluster_centroids_previous)) < 0.001:
                break

predicted_clusters = np.array(cluster_id)
real_clusters = data.to_numpy()[:, -1]

#visualizing data and cluster centroids
centroids_x = [x for x,y,z,_ in cluster_centroids]
centroids_y = [y for x,y,z,_ in cluster_centroids]
centroids_z = [z for x,y,z,_ in cluster_centroids]

fig = plt.figure(figsize = (8, 5))
ax = plt.axes(projection ="3d")

ax.scatter3D(D[:, 0], D[:, 1], D[:, 2], c=predicted_clusters)
ax.scatter3D(centroids_x, centroids_y, centroids_z, marker= '*', c=range(k), s=200)
plt.title('Enhanced KMeans model')
plt.show()

accuracy = (predicted_clusters==real_clusters).sum()/length_D*100
print(f'The accuracy of Enhanced KMeans model:  {accuracy}%')



