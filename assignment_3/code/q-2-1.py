#!/usr/bin/env python
# coding: utf-8

# ### LOGISTIC REGRESSION

# In[10]:


import pandas as pd
import numpy as np
import pprint
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


def sigmoid_funtion(h):
    return 1/(1 + np.exp(-h))


# In[3]:


def train_logistic(X, Y):
    alpha = 0.003
    beta = np.zeros(len(X[0]))
    beta.shape = (len(beta) ,1)
    for i in range(2000):
        H = np.matmul(X, beta)
        G = sigmoid_funtion(H)
        temp = alpha*(np.matmul( X.T , (np.subtract(G, Y))))/len(X)
        beta = beta - temp
    return beta


# In[4]:


def create_Matrix(tf):
    tf = tf.drop(['Chance of Admit '] ,axis = 1)
    tf.insert(0, 'Abs', 1)
    X = tf.values
    return X


# In[5]:


def predict_logistic(beta_matrix, test_matrix):
    H = np.matmul(test_matrix ,beta_matrix)
    return sigmoid_funtion(H)


# In[6]:


def predict_discrete(Y_p ,Y_a , threshold):
    Y_actual = []
    Y_predicted = []
    
    for i in range(len(Y_p)):
        if Y_p[i] > threshold:
            Y_predicted.append(1)
        else:
            Y_predicted.append(0)

        if Y_a[i] > threshold:
            Y_actual.append(1)
        else:
            Y_actual.append(0)
    return Y_predicted , Y_actual
        


# In[19]:


def plot_graph(x_axis , y_axis, y):
    plt.plot(x_axis,y_axis)
    plt.xlabel('Threshold')
    plt.ylabel(y)
    plt.show()


# In[23]:


def logistic_regression():
    frame = pd.read_csv("AdmissionDataset/data.csv").drop(['Serial No.'], axis = 1)
    train_frame = frame.sample(frac = 0.8, random_state = 200)
    test_frame = frame.drop(train_frame.index)
    
    Y_train = train_frame['Chance of Admit '].values
    Y_train.shape = (len(Y_train) , 1)
    Y_test = test_frame['Chance of Admit '].values
    Y_test.shape = (len(Y_test) , 1)
    
    test_frame.drop(['Chance of Admit '], axis = 1)
    train_frame.drop(['Chance of Admit '], axis = 1)
    
    train_frame = (train_frame - train_frame.mean())/train_frame.std()
    test_frame = (test_frame - test_frame.mean())/test_frame.std()
    
    
    X_train = create_Matrix(train_frame)
    beta_matrix = train_logistic(X_train, Y_train)
    
    X_test = create_Matrix(test_frame)
    Y_predicted = predict_logistic(beta_matrix, X_test)
    
    x_axis = []
    accuracy = []
    precision = []
    threshold = 0.5
    for i in range(5):
        print(i)
        Y_predicted, Y_actual = predict_discrete(Y_predicted ,Y_test , threshold)
        x_axis.append(threshold)
        accuracy.append(metrics.accuracy_score(Y_actual , Y_predicted))
        precision.append(metrics.precision_score(Y_actual , Y_predicted))
        print('threshold = ', threshold, metrics.accuracy_score(Y_actual , Y_predicted))
        threshold += 0.1
        
    plot_graph(x_axis , accuracy, 'Accuracy')
    plot_graph(x_axis , precision, 'Precision')
    print(accuracy)
    print(precision)
    
logistic_regression()


# #### Sensitivity measures the proportion of actual positives which are correctly identified.
# 
# #### Specificity measures the proportion of negatives which are correctly identified.
# 
#     Maximizing only sensitivity or specificity is trivial 
#     One criterion is the Youden index: The sum of sensitivity and specificity has to be maximal. However, you can choose a different risk function weighting the trade-off between sensitivity and specificity.
