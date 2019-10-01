#!/usr/bin/env python
# coding: utf-8

# ## Question 3 ONE VS ALL

# In[12]:


import pandas as pd
import numpy as np
import pprint
from sklearn import metrics


# In[13]:


def sigmoid_funtion(h):
    return 1/(1 + np.exp(-h))


# In[118]:


def train_logistic(X, Y):
    alpha = 0.003
    beta = np.zeros(len(X[0]))
    beta.shape = (len(beta) ,1)
    for i in range(300):
        H = np.matmul(X, beta)
        G = sigmoid_funtion(H)
        temp = alpha*(np.matmul( X.T , (np.subtract(G, Y))))/len(X)
        beta = beta - temp
    return beta


# In[150]:


def predict_logistic(beta_matrix, test_matrix):
    H = np.matmul(test_matrix ,beta_matrix.T)
    result = sigmoid_funtion(H)
    predicted = []
    for i in range(len(result)):
        predicted.append(np.where(result == max(result[i][:]))[1][0])
    return predicted


# In[169]:


def one_vs_all():
    frame = pd.read_csv('wine-quality/data.csv')
    op_label = frame['quality']
    frame = frame.drop(['quality'], axis=1)
    
    train_frame = frame.sample(frac = 0.8, random_state=200)
    test_frame = frame.drop(train_frame.index)
        
    train_frame = (train_frame - train_frame.mean()) / train_frame.std()
    test_frame = (test_frame - test_frame.mean()) / test_frame.std()
    train_frame.insert(0, 'Abs', 1)
    
    op_label_train = op_label.sample(frac = 0.8, random_state=200)
    op_label_test = op_label.drop(op_label_train.index)

    X = train_frame.values
    
    theta = []

    Unique_label = [0,1,2,3,4,5,6,7,8,9]
    for val in Unique_label:
        Y = []
        for i in range(len(op_label_train)):
            if op_label[i] == val:
                Y.append(1)
            else:
                Y.append(0)
        Y = np.array(Y)
        Y.shape = (len(Y) ,1)
        th = np.array(train_logistic(X ,Y))
        theta.append(th)
    theta = np.array(theta)
    theta = theta[:,:,0]
    
    test_frame.insert(0 ,'Abs' ,1)
    Test_mat_x = test_frame.values
    Test_mat_y = op_label_test.values
    
    predicted = predict_logistic(theta ,Test_mat_x)
    print("Accuracy = ",metrics.accuracy_score(Test_mat_y ,predicted))
            


# In[170]:


one_vs_all()


# #### Sensitivity measures the proportion of actual positives which are correctly identified.
# 
# #### Specificity measures the proportion of negatives which are correctly identified.
# 
#     Maximizing only sensitivity or specificity is trivial 
#     One criterion is the Youden index: The sum of sensitivity and specificity has to be maximal. However, you can choose a different risk function weighting the trade-off between sensitivity and specificity.
