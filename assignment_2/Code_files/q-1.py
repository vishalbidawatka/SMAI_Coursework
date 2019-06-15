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


# In[3]:


data = pd.read_csv('Iris/Iris.csv',names = ["sepal_length", "sepal_width", "petal_length", "petal_width","class"])


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data['class'].value_counts()


# In[7]:


def plotgraphkacc(listofk,listofacc):
    fig, ax = plt.subplots(figsize = (11,7))  
    ax.plot(listofk,listofacc , 'r')  
    ax.set_xlabel('Values of K')  
    ax.set_ylabel('Accuracy')  
    ax.set_title('Accuracy vs. K values')  


# In[8]:


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
        
    


# In[9]:


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
        
   
        


# # Iris without K-Cross Fold

# In[10]:


train_data = data.sample(frac = 0.8, random_state = 200)
valid_data = data.drop(train_data.index)

attributes = list(data)
attr = deepcopy(attributes)
attr.remove('class')

listofpred = []

avg_accuracy = {}
#k = 9
for nums in range(1,math.floor(np.sqrt(len(train_data))) + 1):
    k = nums
    accuracy = []
    originaly = []
    predy = []
    rightans = 0
    wrongans = 0

    for rowno,testsample in valid_data.iterrows():
        ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
        originaly.append(testsample['class'])
        predy.append(ans)
        #print(ans,testsample['class'])
        if(ans == testsample['class']):
            rightans += 1
        else:
            wrongans += 1
    
    accuracy.append(rightans/(rightans+wrongans))
    print(rightans/(rightans+wrongans))
    avg_accuracy[k] = sum(accuracy) / len(accuracy) 
    listofpred.append(originaly)
    listofpred.append(predy)


# In[11]:


len(train_data)
len(valid_data)


# # Classification report for each k

# In[12]:


for i in range(0,int(len(listofpred)/2)):
    print("K: ",i+1)
    print(sklearn.metrics.classification_report(listofpred[2*i] , listofpred[2*i+1]))


# # Comparison with sklearn KNN

# In[13]:


le = preprocessing.LabelEncoder()
valid_data['class'] = le.fit_transform(valid_data['class'])
train_data['class'] = le.fit_transform(train_data['class'])

for nums in range(1,math.floor(np.sqrt(len(train_data))) + 1):
    model = KNeighborsClassifier(nums,metric= 'euclidean')
    

    accuracy = []
    originaly = []
    predy = []
    
    
    model.fit(train_data.drop('class',axis=1),train_data['class'])
    

    for rowno,testsample in valid_data.iterrows():
        ans = model.predict([testsample.drop('class')])
        originaly.append(testsample['class'])
        predy.append(ans)
        if(ans == testsample['class']):
            rightans += 1
        else:
            wrongans += 1
    
    accuracy.append(rightans/(rightans+wrongans))
    #print(accuracy)
    avg_accuracy[k] = sum(accuracy) / len(accuracy) 
    print("K: ",nums)
    print(sklearn.metrics.classification_report(originaly, predy))


# # Part 2 for IRIS

# ## Iris With K-Cross fold validation for ploting graph w.r.t diff. values of k

# In[14]:




attributes = list(data)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy_kcross = {}
#k = 9
for nums in range(1,10):
    k = nums
    kf = KFold(n_splits = 5,shuffle = True,random_state = 200)
    indices = kf.split(data)
    accuracy = []
    for train_index,valid_index in indices:
        rightans = 0
        wrongans = 0
        train_data = data.loc[train_index]
        valid_data = data.loc[valid_index]
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            if(ans == testsample['class']):
                rightans += 1
            else:
                wrongans += 1
        #print(accuracy)
        accuracy.append(rightans/(rightans+wrongans))
        avg_accuracy_kcross[k] = sum(accuracy) / len(accuracy) 


# In[15]:


print(avg_accuracy_kcross)


# # Plotting graph using k cross validation for different values of K

# In[18]:


plotgraphkacc(avg_accuracy_kcross.keys(),avg_accuracy_kcross.values())


# # Using different distance measures

# In[20]:


distance_measures = ['euclidian','minkowski','manhattan','cosine']
train_data = data.sample(frac = 0.8, random_state = 200)
valid_data = data.drop(train_data.index)

attributes = list(data)
attr = deepcopy(attributes)
attr.remove('class')

listofpred = []

avg_accuracy = {}
#k = 9
for nums in range(1,math.floor(np.sqrt(len(train_data))) + 1):
    print("K: ",nums)
    for distance_met in distance_measures:
        print(distance_met+":",end=" ")
        k = nums
        accuracy = []
        originaly = []
        predy = []
        rightans = 0
        wrongans = 0

        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,distance_met,'class')
            originaly.append(testsample['class'])
            predy.append(ans)
            #print(ans,testsample['class'])
            if(ans == testsample['class']):
                rightans += 1
            else:
                wrongans += 1

        accuracy.append(rightans/(rightans+wrongans))
        print(rightans/(rightans+wrongans))
#         avg_accuracy[k] = sum(accuracy) / len(accuracy) 
#         listofpred.append(originaly)
#         listofpred.append(predy)


# # Part 1 for Robot -1

# ## Robot1 without k cross fold

# In[19]:


data2 = pd.read_csv('RobotDataset/Robot1',sep = ' ',names = ["class", "a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7,8],engine = "python")


# In[20]:


data2 = data2.drop('ID',axis=1)


# In[21]:


data2.info()
data2['class'].value_counts()


# In[43]:


train_data = data2.sample(frac = 0.8, random_state = 200)
valid_data = data2.drop(train_data.index)
attributes = list(data2)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}

for nums in range(1,10):
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
        if (testsample['class'] == decision):
            rcount += 1
            if(decision == 1):
                tp += 1
            else:
                tn += 1
        else:
            wcount += 1
            if(decision == 1):
                fp += 1
            else:
                fn += 1
    print("K:",nums)
    print("accuracy: ",rcount/(rcount+wcount))
    print("true positive: ",tp)
    print("true negative: ",tn)
    print("false positive: ",fp)
    print("false negative: ",fn)
    print("recall: ",(tp)/(tp+fn))
    print("precison: ",(tp)/(tp+fp))
    print("F1 score:",2*(1/((1/((tp)/(tp+fn))) + (1/((tp)/(tp+fp))))))

    accuracy.append(rcount*1.0/(rcount+wcount))
    print()


# # Comparison with scikit learn

# In[23]:


le = preprocessing.LabelEncoder()
valid_data['class'] = le.fit_transform(valid_data['class'])
train_data['class'] = le.fit_transform(train_data['class'])

for nums in range(1,math.floor(np.sqrt(len(train_data))) + 1):
    model = KNeighborsClassifier(nums,metric= 'euclidean')
    

    accuracy = []
    originaly = []
    predy = []
    
    
    model.fit(train_data.drop('class',axis=1),train_data['class'])
    

    for rowno,testsample in valid_data.iterrows():
        ans = model.predict([testsample.drop('class')])
        originaly.append(testsample['class'])
        predy.append(ans)
        if(ans == testsample['class']):
            rightans += 1
        else:
            wrongans += 1
    
    accuracy.append(rightans/(rightans+wrongans))
    #print(accuracy)
    avg_accuracy[k] = sum(accuracy) / len(accuracy) 
    print("K: ",nums)
    print(sklearn.metrics.classification_report(originaly, predy))


# # Part 2 for Robot 1

# ## Robot 1 with K cross fold for ploting graph w.r.t accuracy with different k

# In[42]:


attributes = list(data2)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}

for nums in range(1,10):
    k = nums
    accuracy = []
    kf = KFold(n_splits = 5,shuffle = True,random_state = 200)
    indices = kf.split(data2)
    for train_index,valid_index in indices:
        rightans = 0
        wrongans = 0
        train_data = data2.loc[train_index]
        valid_data = data2.loc[valid_index]
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            if(ans == testsample['class']):
                rightans += 1
            else:
                wrongans += 1
        accuracy.append(rightans*1.0/(rightans+wrongans))
        print(accuracy)
        avg_accuracy[k] = sum(accuracy) / len(accuracy) 


# In[28]:


print(avg_accuracy)


# In[29]:


plotgraphkacc(avg_accuracy.keys(),avg_accuracy.values())


# # Using different distance measures

# In[33]:


distance_measures = ['euclidian','minkowski','manhattan','cosine']
rain_data = data2.sample(frac = 0.8, random_state = 200)
valid_data = data2.drop(train_data.index)
attributes = list(data2)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}

for nums in range(1,10):
    print("K: " + str(nums))
    for distance_met in distance_measures:
        print(distance_met+":",end=" ")
        k = nums
        accuracy = []
        wcount = 0
        rcount  = 0
        tp =0
        tn = 0
        fp = 0
        fn = 0
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,distance_met,'class')
            decision = ans
            if (testsample['class'] == decision):
                rcount += 1
                if(decision == 1):
                    tp += 1
                else:
                    tn += 1
            else:
                wcount += 1
                if(decision == 1):
                    fp += 1
                else:
                    fn += 1

        print("accuracy: ",rcount/(rcount+wcount))
#         print("true positive: ",tp)
#         print("true negative: ",tn)
#         print("false positive: ",fp)
#         print("false negative: ",fn)
#         print("recall: ",(tp)/(tp+fn))
#         print("precison: ",(tp)/(tp+fp))
#         print("F1 score:",2*(1/((1/((tp)/(tp+fn))) + (1/((tp)/(tp+fp))))))

#       #accuracy.append(rcount*1.0/(rcount+wcount))
        #print(accuracy)


# # Part 1 for Robot 2
# ## Robot 2 without k cross fold

# In[34]:


data3 = pd.read_csv('RobotDataset/Robot2',sep = ' ',names = ["class", "a1", "a2", "a3","a4","a5","a6","ID"],usecols = [1,2,3,4,5,6,7,8],engine = "python")


# In[35]:


data3 = data3.drop('ID',axis=1)


# In[41]:


train_data = data3.sample(frac = 0.8, random_state = 200)
valid_data = data3.drop(train_data.index)
attributes = list(data3)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}

for nums in range(1,10):
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
        if (testsample['class'] == decision):
            rcount += 1
            if(decision == 1):
                tp += 1
            else:
                tn += 1
        else:
            wcount += 1
            if(decision == 1):
                fp += 1
            else:
                fn += 1
    print("K:",nums)
    print("accuracy: ",rcount/(rcount+wcount))
    print("true positive: ",tp)
    print("true negative: ",tn)
    print("false positive: ",fp)
    print("false negative: ",fn)
    print("recall: ",(tp)/(tp+fn))
    print("precison: ",(tp)/(tp+fp))
    print("F1 score:",2*(1/((1/((tp)/(tp+fn))) + (1/((tp)/(tp+fp))))))

    accuracy.append(rcount*1.0/(rcount+wcount))
    print()


# # Comparison with sklearn

# In[44]:


le = preprocessing.LabelEncoder()
valid_data['class'] = le.fit_transform(valid_data['class'])
train_data['class'] = le.fit_transform(train_data['class'])

for nums in range(1,math.floor(np.sqrt(len(train_data))) + 1):
    model = KNeighborsClassifier(nums,metric= 'euclidean')
    

    accuracy = []
    originaly = []
    predy = []
    
    
    model.fit(train_data.drop('class',axis=1),train_data['class'])
    

    for rowno,testsample in valid_data.iterrows():
        ans = model.predict([testsample.drop('class')])
        originaly.append(testsample['class'])
        predy.append(ans)
        if(ans == testsample['class']):
            rightans += 1
        else:
            wrongans += 1
    
    accuracy.append(rightans/(rightans+wrongans))
    #print(accuracy)
    avg_accuracy[k] = sum(accuracy) / len(accuracy) 
    print("K: ",nums)
    print(sklearn.metrics.classification_report(originaly, predy))


# # Part 2 for Robot2

# ## Robot 2 with k cross fold validation for plotting graph w.r.t. different k

# In[45]:


attributes = list(data3)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}
# k = 22 #robo1
# k = 29# robo2
for nums in range(1,10):
    k = nums
    accuracy = []
    kf = KFold(n_splits = 5,shuffle = True,random_state = 200)
    indices = kf.split(data3)
    for train_index,valid_index in indices:
        rightans = 0
        wrongans = 0
        train_data = data3.loc[train_index]
        valid_data = data3.loc[valid_index]
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,'euclidian','class')
            if(ans == testsample['class']):
                rightans += 1
            else:
                wrongans += 1
        accuracy.append(rightans*1.0/(rightans+wrongans))
        print(accuracy)
        avg_accuracy[k] = sum(accuracy) / len(accuracy) 


# In[75]:


print(avg_accuracy)


# In[46]:


plotgraphkacc(avg_accuracy.keys(),avg_accuracy.values())


# # Different distance measures for robot2

# In[47]:


distance_measures = ['euclidian','minkowski','manhattan','cosine']
rain_data = data3.sample(frac = 0.8, random_state = 200)
valid_data = data3.drop(train_data.index)
attributes = list(data3)
attr = deepcopy(attributes)
attr.remove('class')

avg_accuracy = {}

for nums in range(1,10):
    print("K: " + str(nums))
    for distance_met in distance_measures:
        print(distance_met+":",end=" ")
        k = nums
        accuracy = []
        wcount = 0
        rcount  = 0
        tp =0
        tn = 0
        fp = 0
        fn = 0
        for rowno,testsample in valid_data.iterrows():
            ans = getkneighbours(train_data,testsample,k,attr,distance_met,'class')
            decision = ans
            if (testsample['class'] == decision):
                rcount += 1
                if(decision == 1):
                    tp += 1
                else:
                    tn += 1
            else:
                wcount += 1
                if(decision == 1):
                    fp += 1
                else:
                    fn += 1

        print("accuracy: ",rcount/(rcount+wcount))
#         print("true positive: ",tp)
#         print("true negative: ",tn)
#         print("false positive: ",fp)
#         print("false negative: ",fn)
#         print("recall: ",(tp)/(tp+fn))
#         print("precison: ",(tp)/(tp+fp))
#         print("F1 score:",2*(1/((1/((tp)/(tp+fn))) + (1/((tp)/(tp+fp))))))

#       #accuracy.append(rcount*1.0/(rcount+wcount))
        #print(accuracy)

