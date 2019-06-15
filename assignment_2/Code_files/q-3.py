#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# # Part 1

# In[27]:


import pandas as pd
import random
import numpy as  np
import math
from copy import deepcopy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import operator
iterations = 1000
learning_rate = 0.009


# In[47]:


def plotgraphs(iterations,weights, ind):
    fig, ax = plt.subplots(figsize = (7,9))  
    ax.plot(np.arange(iterations),weights , 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('wts')  
    ax.set_title('wt:' + str(ind) + ' vs. Training Epoch')  


# In[28]:


data = pd.read_csv('../input_data/AdmissionDataset/data.csv')


# In[29]:


data = data.drop('Serial No.',axis = 1)


# ## Normalizing Data

# In[30]:


#data= data.apply(lambda x: x/x.max(), axis=0)


# In[31]:


data.head()


# In[32]:


train_data = data.sample(frac = 0.8, random_state = 200)
valid_data = data.drop(train_data.index)
# split = int(0.8*data.shape[0])
# train_data = data[:split]
# valid_data  = data[split:]


# In[33]:


train_data.head()


# In[34]:


datamatrix  = train_data.iloc[:,0:7]
print(datamatrix.head())
trainmean = datamatrix.mean()
trainstd = datamatrix.std()
datamatrix = (datamatrix - datamatrix.mean())/datamatrix.std()
datamatrix.insert(loc=0, column ='ones', value=1)
actual_values = train_data['Chance of Admit '].values
actual_y_values = valid_data['Chance of Admit '].values
actual_values = np.reshape(actual_values,(360,1))
datamatrix = np.array(datamatrix)
print(datamatrix.shape)
weights = np.zeros([1,8])
print(weights)


# In[36]:


def jcost(datamatrix,actual,wts):
    predicted = (datamatrix @ wts.T)
    #print(predicted)
    error_vector = np.power(actual - predicted,2)
    #print(error_vector)
    #print(len(datamatrix))
    return np.sum(error_vector)/(2*len(datamatrix))


# In[37]:


jcost(datamatrix,actual_values,weights)


# In[38]:


weightstho = []


# In[39]:


def gd(datamatrix,actual_values,wts):
    cost = []
    for i in range(iterations):
        bias  =  (( (datamatrix @ wts.T) - actual_values).T)@datamatrix 
        weightstho.append(wts)

        wts = wts - ((learning_rate/(len(datamatrix))))*bias
        #print(wts)
        cost.append(jcost(datamatrix,actual_values,wts))
    return cost, wts


# In[40]:


valid_data.head()


# In[41]:


valid_data = valid_data.drop('Chance of Admit ',axis = 1 )
valid_data = (valid_data - trainmean)/trainstd
valid_data.insert(loc=0, column ='ones', value=1)
valid_data.head()


# In[42]:


valid_data.head()
valid_data = valid_data.iloc[:,0:8]
valid_data.head()


# # Part 2 and Part 3

# In[43]:


cost,wts = gd(datamatrix,actual_values,weights)
pred = valid_data.dot(wts.T)
rms = 0
abserror = 0
for i,j in zip(pred,actual_y_values):
    rms += math.pow(j-i,2)
    abserror += abs(j-i)


# In[63]:


print("Root mean square :",rms)
print("Absolute Error: ",abserror)


# In[53]:


print(wts)


# # Error vs iterations graph

# In[55]:


fig, ax = plt.subplots(figsize = (14,9))  
ax.plot(np.arange(iterations),cost , 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  


# # R2 Score

# In[46]:


from sklearn.metrics import r2_score
r2_score(actual_y_values,pred)


# In[54]:


wts


# # Plotting graph of wieghts vs iterations

# In[62]:


for j in range(len(wts[0])):
    wt = []
    print("wieght: "+str(j))
    for i in range(iterations):
        wt.append(weightstho[i][0][j])
    fig, ax = plt.subplots()  
    ax.plot(np.arange(iterations),wt , 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('wts')  
    ax.set_title('wt:' + str(j) + ' vs. Training Epoch') 
    plt.show()
    

