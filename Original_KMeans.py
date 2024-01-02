import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits import mplot3d

class KmeansClustering():
    
    def __init__(self, k = 3):
        self.k = k

    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))


    def fit(self, X, max_iterations = 200):
        #random initializing first centroids
        self.centroids =  np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                            size = (self.k, X.shape[1]))
       
        for _ in range(max_iterations):

            y = []
            #finding minimum distance between each datapoint and centroids 
            for data_point in X:
                distances = KmeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num  = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            #finding indices of datapoints belonging to certain clusters
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y==i))
            
            #arranging new centers for datapoints
            cluster_centers = []

            for i, indices in enumerate(cluster_indices):

                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            
            #if the change in coordinates is small, break
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y
#data loading and preprocessing 
data = pd.read_csv('iris.csv')

data['variety'] = data['variety'].replace('Setosa',1)
data['variety'] = data['variety'].replace('Versicolor', 0)
data['variety'] = data['variety'].replace('Virginica', 2)

D = data.to_numpy()[:,:-1]
real_clusters = data.to_numpy()[:, -1]
length_D = len(D)

kmeans = KmeansClustering(k=3)
labels = kmeans.fit(D)

#data visualization
fig = plt.figure(figsize = (8, 5))
ax = plt.axes(projection ="3d")

ax.scatter3D(D[:, 0], D[:, 1], D[:,2], c=labels)
ax.scatter3D(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:,2], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.title('Original KMeans model')
plt.show()

accuracy = (labels==real_clusters).sum()/length_D*100
print(f'The accuracy of Original KMeans model:  {accuracy}%')
