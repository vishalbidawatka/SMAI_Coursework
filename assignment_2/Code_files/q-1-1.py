#!/usr/bin/env python
# coding: utf-8

# # Part 1 for IRIS

# In[2]:


import pandas as pd
import random
import numpy as  np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import sklearn
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import operator
import sys


# In[3]:

def distancemeasure(row1,row2,attributes,metric):
    distance = 0.0
    if metric == "euclidian":
        for attr in attributes:
            distance = distance + abs(row1[attr] - row2[attr])**2
        return np.sqrt(distance)
    if metric == "manhattan":
        for attr in attributes:
            distance +=  abs(row1[attr] - row2[attr])
        return distance
    if metric == "minkowski":
        p = 1.5
        for attr in attributes:
            distance += np.power(abs(row1[attr] - row2[attr]),p)
        return np.power(distance,(1.0)/p)
    if metric == "cosine":
        xisquare = 0.0
        yisquare = 0.0
        for attr in attributes:
            xisquare += row1[attr]**2
            yisquare += row2[attr]**2
        distance = 0.0
        for attr in attributes:
            distance += row1[attr]*row2[attr]
        
        return 1 - ((distance)/(np.sqrt(xisquare)*np.sqrt(yisquare)))


def getkneighbours(train_data,test_sample,k,attributes,metric,outputclass):
    distancelist = []
    for rowno,train_sample in train_data.iterrows():
        distancelist.append([distancemeasure(train_sample,test_sample,attributes,metric),train_sample[outputclass]])
    distancelist.sort()
    dictofresults = {}
    for nums in range(k):
        if distancelist[nums][1] in dictofresults.keys():
            dictofresults[distancelist[nums][1]] += 1
        else:
            dictofresults[distancelist[nums][1]] = 1
    #print(distancelist)
    #print(dictofresults)
    #print(max(dictofresults.items(), key=operator.itemgetter(1))[0],"selected")
    return max(dictofresults.items(), key=operator.itemgetter(1))[0]
            

def general_method(datasetselector , testfile):
    if datasetselector == 0:

        data = pd.read_csv('Iris/Iris.csv',names = ["sepal_length", "sepal_width", "petal_length", "petal_width","class"])
        train_data = data
        valid_data =  pd.read_csv(testfile,names = ["sepal_length", "sepal_width", "petal_length", "petal_width"])

        attributes = list(data)
        attr = deepcopy(attributes)
        attr.remove('class')

        listofpred = []

        avg_accuracy = {}
        #k = 9
        nums = 9
        k = nums
        accuracy = []
        originaly = []
        predy = []
        rightans = 0
        wrongans = 0

        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            predy.append(ans)
            print(ans)



    if datasetselector == 1:
        data2 = pd.read_csv('RobotDataset/Robot1',sep = ' ',names = ["class", "a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7,8],engine = "python")
        data2 = data2.drop('ID',axis=1)

        # In[43]:


        train_data = data2
        valid_data = pd.read_csv(testfile,sep = ' ',names = ["a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7],engine = "python")
        valid_data = valid_data.drop('ID',axis=1)

        attributes = list(data2)
        attr = deepcopy(attributes)
        attr.remove('class')

        avg_accuracy = {}

        nums = 9
        k = nums
        accuracy = []
        wcount = 0
        rcount  = 0
        tp =0
        tn = 0
        fp = 0
        fn = 0
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            decision = ans
            print(ans)

    # # Part 1 for Robot 2
    # ## Robot 2 without k cross fold

    # In[34]:

    if datasetselector == 2:
        data3 = pd.read_csv('RobotDataset/Robot2',sep = ' ',names = ["class", "a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7,8],engine = "python")


        # In[35]:


        data3 = data3.drop('ID',axis=1)
        train_data = data3
        valid_data = pd.read_csv(testfile,sep = ' ',names = ["a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7],engine = "python")
        valid_data = valid_data.drop('ID',axis=1)
        # In[41]:



        attributes = list(data3)
        attr = deepcopy(attributes)
        attr.remove('class')

        avg_accuracy = {}

        nums = 9
        k = nums
        accuracy = []
        wcount = 0
        rcount  = 0
        tp =0
        tn = 0
        fp = 0
        fn = 0
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            decision = ans
            print(ans)



if __name__ == "__main__":
   general_method(int(sys.argv[1]) , sys.argv[2])
