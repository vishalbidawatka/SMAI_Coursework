#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import random
import numpy as  np
import math
import sys
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import operator
iterations = 1000
learning_rate = 0.009


def jcost(datamatrix,actual,wts):
    predicted = (datamatrix @ wts.T)
    #print(predicted)
    error_vector = np.power(actual - predicted,2)
    #print(error_vector)
    #print(len(datamatrix))
    return np.sum(error_vector)/(2*len(datamatrix))


def gd(datamatrix,actual_values,wts):
    cost = []
    for i in range(iterations):
        bias  =  (( (datamatrix @ wts.T) - actual_values).T)@datamatrix 
        #print(bias)
        wts = wts - ((learning_rate/(len(datamatrix))))*bias
        #print(wts)
        cost.append(jcost(datamatrix,actual_values,wts))
    return cost, wts






def general_method(testfile):
    data = pd.read_csv('AdmissionDataset/data.csv')
    valid_data = pd.read_csv(testfile)


    data = data.drop('Serial No.',axis = 1)
    valid_data = valid_data.drop('Serial No.',axis = 1)




    train_data = data.sample(frac = 0.8, random_state = 200)



    datamatrix  = train_data.iloc[:,0:7]
    print(datamatrix.head())
    trainmean = datamatrix.mean()
    trainstd = datamatrix.std()
    datamatrix = (datamatrix - datamatrix.mean())/datamatrix.std()
    datamatrix.insert(loc=0, column ='ones', value=1)
    actual_values = train_data['Chance of Admit '].values
    #actual_y_values = valid_data['Chance of Admit '].values
    actual_values = np.reshape(actual_values,(360,1))
    datamatrix = np.array(datamatrix)
    print(datamatrix.shape)
    weights = np.zeros([1,8])
    print(weights)




    print(datamatrix.shape)
    print(weights.shape)
    print(actual_values.shape)


    # In[48]:


    valid_data.head()


    # In[49]:



    valid_data = (valid_data - trainmean)/trainstd
    valid_data.insert(loc=0, column ='ones', value=1)
    print("here ")
    print(valid_data.head())
    print(list(valid_data))

    


    # In[51]:


    cost,wts = gd(datamatrix,actual_values,weights)
    pred = valid_data.dot(wts.T)
    leng = 0
    for i,x in pred.iterrows():
        leng+=1
        print(x[0])
    #print(leng)





if __name__ == "__main__":
   general_method(sys.argv[1])

