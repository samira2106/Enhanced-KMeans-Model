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
# Results and Evaluation
Considering the results that the authors of the reasearch paper obtained, I could get the accuracy 89.33% by comparing the dataset's real clusters labels with assigned cluster labels and the accuracy stay the same with each run of the program. Compared to Original KMeans model results, the initialization of initial centroids is random or requires the initial input by the user, so the model results with different clustering of data points and accuracy every time the program is executed. 

3D visualizations of models results are shown below:

<img src="https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/5bef07c5-5297-4b37-9100-324ef49186bb)https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/5bef07c5-5297-4b37-9100-324ef49186bb" width="450" hight=400 />
<img src="https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/541de817-e56f-4f02-b87d-2e6afe523822)https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/541de817-e56f-4f02-b87d-2e6afe523822" width="400" hight=400 /> 
<img src="https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/91b29db4-ea17-4cf2-8c46-99aa28be2e32" width="400" hight=400> 

The last 2 visualizations show the inconsistency of clustering in Original KMeans model.
# Future Improvements
The main problem stated in the research paper was k values definition and, for solving this problem, I used an Elbow Method. It involves running the k-means algorithm for various k values, calculating the sum of squared distances (inertia) for each k, and then plotting these values against k. The optimal k is identified at the "elbow" point in the plot. 

Below the plot of Elbow Method can be seen for the 'Iris' dataset. It can be seen that the best number of clusters would be 3 or 4. As the increase in k values result in a slight decrease of inertia, the smaller k value is chosen for less complexity of the program.

<img src="https://github.com/samira2106/Enhanced-KMeans-Model/assets/154353012/1337fbd6-e9c1-4d1d-b506-c68f49eeefbc" width="600" hight=600> 

Despite obtaining the accuracy of the model, I could not achieve the improvements in time efficiency due to not ultimately optimized code. This may be compensated further as my programming optimization skills will improve and I will make some changes to the current program. 

# Conclusion
In summary, the enhanced k-means algorithm emerges as a notable enhancement, mitigating the challenges associated with random centroid initialization. While the assessment of time complexity is still ongoing, the primary emphasis remains on achieving improved accuracy and implementing Elbow Method for k value definition problem.
