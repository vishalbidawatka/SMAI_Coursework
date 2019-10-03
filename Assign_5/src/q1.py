#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np
import pandas as pd
from numpy.linalg import eig
import math
from collections import Counter
import random as rd
from collections import Counter
import matplotlib.pyplot as plt
import copy
import csv


# In[80]:


def der_tanh(z):
    return (1 - (np.tanh(z)**2))


# In[30]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[5]:


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


# In[6]:


def softMax(Z):
    E = np.exp(Z - np.max(Z, axis = 1, keepdims=True))
    S = np.sum(E, axis = 1, keepdims= True)
    A = E/S
#     print(A)
    return A


# In[86]:


def reLu(z):
    z1 = copy.deepcopy(z)
    z1[z1 < 0] = 0
    return z1


# In[88]:


def der_reLu(z):
    z1 = copy.deepcopy(z)
    z1[z1 <= 0] = 0
    z1[z1 > 0] = 1
    return z1


# In[97]:


def createBatch(epoch):
    for i  in range(epoch):
        batchSize = 100
        start = 0
        end = batchSize
        while(end < len(training_data)):
            batch = training_data.iloc[start:end]
            output = neuralNet.forwardPropagation(batch)
            neuralNet.backwardPropagation(batch, modifiedTrainLabel[start:end,:])
            start = start + batchSize
            end = end + batchSize


# In[8]:


def modifyLabelVector(x, trainLabel):
    columns = len(np.unique(trainLabel))
    rows = x.shape[0]
    modifiedLabels = np.zeros((rows, columns))

    for i in range(rows):
        modifiedLabels[i, trainLabel[i]] = 1
        
    return modifiedLabels


# In[127]:


class NeuralNet:
    def __init__(self, x, y, hiddenLayers):
        self.initialLayerSize = x.shape[1]
        self.outputLayerSize = len(y.unique())
        
        self.hiddenLayersCount = len(hiddenLayers)
        
        self.Weights = [0] * (self.hiddenLayersCount + 1)
        self.Z = [0] * (self.hiddenLayersCount + 1)
        self.ActivateFunc = [0] * (self.hiddenLayersCount + 1)
        
        for i in range(len(hiddenLayers) + 1):
            if i == 0:
                self.Weights[i] = np.random.randn(self.initialLayerSize, hiddenLayers[0])
            elif i == self.hiddenLayersCount:
                self.Weights[i] = np.random.randn(hiddenLayers[self.hiddenLayersCount - 1], self.outputLayerSize)
            else:
                self.Weights[i] = np.random.randn(hiddenLayers[i - 1], hiddenLayers[i])
                
#             print("Weight ", i, " ", self.Weights[i].shape)
                
        
    def forwardPropagation(self, X):
        self.Z[0] = np.dot(X, self.Weights[0])
        self.ActivateFunc[0] = np.tanh(self.Z[0])
        
        for i in range(1, self.hiddenLayersCount + 1):
            self.Z[i] = np.dot(self.ActivateFunc[i - 1], self.Weights[i])
            self.ActivateFunc[i] = np.tanh(self.Z[i])
            
        self.sm = softMax(self.Z[self.hiddenLayersCount])
        return self.sm
        
    def backwardPropagation(self, X, Y):
        
        delta = [None] * (self.hiddenLayersCount + 1)
        dJdW = [None] * (self.hiddenLayersCount + 1)
        
#         print("Active Func Shape: ", self.ActivateFunc[self.hiddenLayersCount - 1].shape)
        delta[self.hiddenLayersCount] = -(Y - self.sm) * der_tanh(self.ActivateFunc[-1])
        dJdW[self.hiddenLayersCount] = np.dot(self.ActivateFunc[self.hiddenLayersCount - 1].T, delta[self.hiddenLayersCount])
        
#         print("dj/dw: ", self.hiddenLayersCount, " ", dJdW[self.hiddenLayersCount])
        
        i = self.hiddenLayersCount - 1
        
        while i >= 0:
            if i == 0:
                delta[i] = np.dot(delta[i + 1], self.Weights[i + 1].T) * der_tanh(self.ActivateFunc[i])
                dJdW[i] = np.dot(X.T, delta[i])
            else:
                delta[i] = np.dot(delta[i + 1], self.Weights[i + 1].T) * der_tanh(self.ActivateFunc[i])
                dJdW[i] = np.dot(self.ActivateFunc[i - 1].T, delta[i])
                
                
            i -= 1
            
        i = self.hiddenLayersCount
        while i >= 0 :
            self.Weights[i] = self.Weights[i] - 0.01 * dJdW[i]
        
#             print(self.Weights[i])
            i -= 1
            
    def predict(self, data):
        return self.forwardPropagation(data)


# In[10]:


data = pd.read_csv('Apparel/apparel-trainval.csv')

training_data, test_data = np.split(data, [int(0.8*len(data))])

trainLabel = training_data['label']
testLabel = test_data['label']

training_data = training_data.drop(columns='label')
test_data = test_data.drop(columns='label')


# In[11]:


training_data=training_data.astype('float64')
means = np.mean(training_data, axis = 0)
stdDev = np.std(training_data, axis = 0)
training_data = (training_data - means) / stdDev


# In[12]:


test_data=test_data.astype('float64')
testMeans = np.mean(test_data, axis = 0)
testStdDev = np.std(test_data, axis = 0)
test_data = (test_data - testMeans) / testStdDev


# In[13]:


trainLabelValues = trainLabel.values
modifiedTrainLabel = modifyLabelVector(training_data, trainLabelValues)


# In[14]:


testLabelValues = testLabel.values
modifiedTestLabel = modifyLabelVector(test_data, testLabelValues)


# In[128]:


neuralNet = NeuralNet(training_data, trainLabel, [128])
epoch = 20
epochs.append(epoch)
createBatch(epoch)


# In[77]:


activAccuracy = []


# In[129]:


prediction = neuralNet.forwardPropagation(test_data)
print(len(prediction))
correct_cnt = 0
for i in range(len(prediction)):
    temp1 = np.argmax(modifiedTestLabel[i])
    temp2 = np.argmax(prediction[i])
    if(temp1 == temp2):
        correct_cnt += 1
# accuracies.append(correct_cnt * 1.0 / len(prediction))
# epochAcc.append(correct_cnt * 1.0 / len(prediction))
print("Accuracy:", correct_cnt * 1.0 / len(prediction))


# In[85]:


activAccuracy


# # Accuracy vs Number of Hidden Layers

# In[40]:


accuracies = []
layers = []


# In[72]:


layers.append(neuralNet.hiddenLayersCount)


# In[74]:


plt.figure()
plt.plot(layers, accuracies)
plt.show()


# # Accuracy vs No. of Epochs

# In[103]:


epochs = []
epochAcc = []


# In[124]:


epochAcc


# In[125]:


plt.figure()
plt.plot(epochs, epochAcc)
plt.show()


# In[144]:


test = pd.read_csv('apparel-test.csv')

test = test.astype('float64')
means = np.mean(test, axis = 0)
stdDev = np.std(test, axis = 0)
test = (test - means) / stdDev

result = []

prediction = neuralNet.forwardPropagation(test)
for i in range(len(prediction)):
    result.append(np.argmax(prediction[i]))
    
np.savetxt('2018202021_prediction.csv', result, delimiter=",")

