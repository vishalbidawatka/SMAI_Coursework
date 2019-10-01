#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import q_1_1
from random import randint
from numpy import linalg
import copy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


# In[69]:


def nearest_cluster(row, centroids):
    min_dis = float('Inf')
    centroid = 0
    for i in range(len(centroids)):
        dis = np.sqrt(np.sum((row - centroids[i]) ** 2))
        if min_dis > dis:
            min_dis = dis
            centroid = i + 1
    return centroid


# In[70]:


def shift_cluster_centres(frame ,clusters):
    K = np.unique(clusters)
    new_clusters = []
    for i in range(len(K)):
        p = np.mean(frame[clusters == K[i]] ,axis = 0)
        new_clusters.append(p)
    new_clusters = np.array(new_clusters)
    return new_clusters


# In[71]:


def clusters_purity(clusters, labels, K):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    clst = np.zeros((K, len(unique_labels)))
    for i in range(len(labels)):
        cNo = int(clusters[i])
        label = int(np.nonzero(unique_labels == labels[i])[0][0])
        clst[cNo - 1][label] += 1
    
    avg_purity = 0
    rowSum = clst.sum(axis = 1)
    for i in range(len(clst)):
        for j in range(len(clst[0])):
            clst[i][j] = clst[i][j] / rowSum[i]
        purity = clst.max(axis = 1)[i]
        cluster_no = np.nonzero(clst[i] == purity)[0]
        temp = rowSum[i]/len(labels)
        temp = temp*purity
        avg_purity += temp
        
        
        print('The Purity of Cluster Number',i + 1,'is ',purity,' val ',unique_labels[cluster_no])
    return avg_purity
        


# ### K-Means Clustering

# In[72]:


def k_means_clustering(K):
    frame = pd.read_csv('intrusion detection/data.csv').drop(['xAttack'] ,axis = 1)
    frame = q_1_1.PCA(frame)

    old_centroids = []
    new_centroids = []
    
    for i in range(K):
        index = randint(1, len(frame))
        old_centroids.append(frame[index])

    old_centroids = np.array(old_centroids)
    new_centroids = np.zeros(old_centroids.shape)

    clusters = np.zeros(len(frame))
    error = linalg.norm(old_centroids - new_centroids)
    itr = 0
    while error != 0:
        itr += 1
        for i in range(len(frame)):
            clusters[i] = nearest_cluster(frame[i], old_centroids)
        new_centroids = copy.deepcopy(old_centroids)
        
        old_centroids = shift_cluster_centres(frame, clusters)
        error = linalg.norm(old_centroids - new_centroids)
    print('Number of Iterations = ',itr)
    labels = pd.read_csv('intrusion detection/data.csv', usecols = ['xAttack'])
    avg_purity = clusters_purity(clusters, labels, K)
    return avg_purity
#     return labels, clusters


# In[73]:


k_avg_purity = k_means_clustering(5)


# In[74]:


print(k_avg_purity)


# ### GMM 

# In[75]:



def gaussian_mixture_model(G):
    frame = pd.read_csv('intrusion detection/data.csv')
    labels = frame['xAttack']
    frame = frame.drop(['xAttack'], axis = 1)
    frame = q_1_1.PCA(frame)
    
    gmm = GaussianMixture(n_components=5)
    clusters = gmm.fit_predict(frame)
    avg_purity = clusters_purity(clusters, labels, 5)
    return avg_purity


# In[76]:


gmm_avg_purtiy = gaussian_mixture_model(5)


# In[77]:


gmm_avg_purtiy


# ### Aglomerative Clustering

# In[78]:


def agglomerative_clustering(G):
    frame = pd.read_csv('intrusion detection/data.csv')
    labels = frame['xAttack']
    frame = frame.drop(['xAttack'], axis = 1)
    frame = q_1_1.PCA(frame)
    
    clustering = AgglomerativeClustering(n_clusters = 5, linkage='single')
    clusters = clustering.fit_predict(frame)
    avg_purity = clusters_purity(clusters, labels, G)
    return avg_purity


# In[79]:


agg_avg_purity = agglomerative_clustering(5)


# In[80]:


print(agg_avg_purity)


# In[84]:


labels = 'K-Means', 'GMM', 'Agglomerative Clustering'
sizes = [k_avg_purity,gmm_avg_purtiy,agg_avg_purity]
colors = ['red', 'green', 'silver']
explode = (0.05, 0.05, 0.05)
 
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()

