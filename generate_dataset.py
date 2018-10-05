# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:07:44 2018

@author: a232khan

"""


from __future__ import division

import pickle, gzip
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

D = 4 # Dimension of 2D image
N_CLUSTERS = D*D # Number of classes
N_DATA = 1000 * N_CLUSTERS # Number of sample images
N_TRAIN = int(round(2.0/3.0 * N_DATA))
dataset_prefix = "[SIM]"

seed = np.random.randint(10000)
print("Seed = %d"  %(seed))



def plot_image(image, comment):
    normed_image = (255 * image / np.amax(image))
    plt.ioff()
    plt.figure()
    plt.grid(True)
    plt.imshow(normed_image, cmap="gray", interpolation="None")
    plt.savefig('Data/%s %s' %(dataset_prefix, comment))
    plt.close()


# Check if directory exists
if not os.path.exists('Data'):
    os.makedirs('Data')
    
    
out_path = os.path.join(os.getcwd(), 'Data')
for f in os.listdir(out_path):
    if re.search(dataset_prefix, f):
        os.remove(os.path.join(out_path, f))

         
data = []
cluster_data = [] # Used for clustering
archetypes = []
labels = []
data_centroids = []

## Method 1: Uniform random sampling, then cluster
for i in range(N_DATA):
    # Maybe add some structure? This is a uniform point cloud unlike MNIST right now
    curr_image = np.random.random((D*D))
    
    #normed_image = (255 * curr_image / np.amax(curr_image))#.astype(int)
    cluster_data += [curr_image]
   
## Label using archetypes (k-means)
##nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data, archetypes)
##distances, labels = nbrs.kneighbors(data)    
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(cluster_data)    
#cluster_labels = kmeans.predict(cluster_data)
archetypes = kmeans.cluster_centers_

### Method 2: Uniform random sampling of cluster centroids
#for i in range(N_CLUSTERS):
#    archetypes += [np.random.random(D*D)]

intra_distances = 0.0

cluster_index = -1
for i in range(N_DATA):
    # Even split among clusters
    if i % (N_DATA / N_CLUSTERS) == 0:
        cluster_index += 1
    data += [ np.random.normal(archetypes[cluster_index], scale=np.ones(D*D) * 0.1) ]
    data_centroids += [ archetypes[cluster_index]]
    labels += [cluster_index]
    #reshaped = np.reshape(data[i], (D,D))
    #plot_image(reshaped, "Data Archetype %d %d" %(cluster_index, i))
    # Distance to cluster centroid
    intra_distances += np.linalg.norm(data[i] - archetypes[cluster_index])
    
print(intra_distances/N_DATA)    

for i in range(len(archetypes)):
    image = np.reshape(archetypes[i], (D,D))
    plot_image(image, "Archetype %d" %(i))
    
## Shuffle data
data_arr = np.asarray(data)
data_centroids_arr = np.asarray(data_centroids)
labels_arr = np.asarray(labels)
s = np.arange(data_arr.shape[0])
np.random.shuffle(s)
data = data_arr[s]
labels = labels_arr[s]
data_centroids = data_centroids_arr[s]

# Split dataset into training and test sets
data_train = data[:N_TRAIN]
data_test = data[N_TRAIN:]
data_centroids_train = data_centroids[:N_TRAIN]
data_centroids_test = data_centroids[N_TRAIN:]
labels_train = labels[:N_TRAIN]
labels_test = labels[N_TRAIN:]
to_dump = archetypes, (data_train, labels_train, data_centroids_train), (data_test, labels_test, data_centroids_test) 

f = open(os.path.join(out_path, "%s.data" %(dataset_prefix) ), 'wb')
pickle.dump(to_dump, f, protocol=2)
f.close()
