#!/usr/bin/env python
# coding: utf-8

# ### LOGISTIC REGRESSION

# In[26]:


import pandas as pd
import numpy as np
import pprint
from sklearn import metrics
import matplotlib.pyplot as plt
from operator import itemgetter


# In[3]:


def sigmoid_funtion(h):
    return 1/(1 + np.exp(-h))


# In[4]:


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


# In[5]:


def create_Matrix(tf):
    tf = tf.drop(['Chance of Admit '] ,axis = 1)
    tf.insert(0, 'Abs', 1)
    X = tf.values
    return X


# In[6]:


def predict_logistic(beta_matrix, test_matrix):
    H = np.matmul(test_matrix ,beta_matrix)
    return sigmoid_funtion(H)


# In[7]:


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
        


# In[10]:


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
        
    print(accuracy)
    print(precision)
    
logistic_regression()


# ### K-NN

# In[23]:


def euclidean_dis(train_frame, row):
    dist = []
    for i,r in train_frame.iterrows():
        sum = 0
        for col in train_frame:
            if col != "Chance of Admit ":
                sum += ((row[col] - r[col]) ** 2)
        sum = np.sqrt(sum)
        temp = [sum, r["Chance of Admit "]]
        dist.append(temp)
    return dist


# In[24]:


def predict_admission(train_frame, test_frame, K):
    n_of_correct = 0
    prediction_list = []
    dis_list = []
    for i,r in test_frame.iterrows():
        dis_list = euclidean_dis(train_frame, r)

        dis_list = sorted(dis_list, key = itemgetter(0))

        count_dict = {}
        for x in train_frame["Chance of Admit "].unique():
            count_dict[x] = 0
            
        for k in range(K):
            count_dict[dis_list[k][1]] += 1
        
        predicted = max(count_dict.items(), key = itemgetter(1))[0]
        prediction_list.append(predicted)
        if predicted == r["Chance of Admit "]:
            n_of_correct += 1
    return (n_of_correct)/len(test_frame)


# In[30]:


def admission():
    frame_admission = pd.read_csv("AdmissionDataset/data.csv").drop(['Serial No.'], axis = 1)
    op = 'Chance of Admit '
    frame.loc[frame[op] > 0.5, op] = 1
    frame.loc[frame[op] <= 0.5, op] = 0
#     frame['Chance of Admit '][frame['Chance of Admit '] <= 0.5] = 0
        
    train_frame_admission = frame_admission.sample(frac = 0.8, random_state = 200)
    test_frame_admission = frame_admission.drop(train_frame_admission.index)
    
    predict_dict = {}
    for i in range(5, 20):
        predict_dict[i] = predict_admission(train_frame_admission, test_frame_admission, i)
    
    pprint.pprint(predict_dict)
#     pprint.pprint(test_frame_admission['Chance of Admit '])

admission()

