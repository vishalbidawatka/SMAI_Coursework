#!/usr/bin/env python
# coding: utf-8

# # Assignment 1
# 
# ### we have to build a decision tree from scratch using various calclulations
# 

# ## Part 1 (30 points)
# Train decision tree only on categorical data. Report precision,recall, f1 score and accuracy.

# #### Imporitng Libraries

# In[1]:


import pandas as pd
import numpy
import random
import math
from sklearn.model_selection import train_test_split
eps = numpy.finfo(float).eps
from copy import deepcopy


# #### Reading data from CSV and it's analysis

# In[2]:


data = pd.read_csv('input_data/train.csv')
data.head()


# In[3]:


data.info()


# #### So, there is no missing values in the dataset and hence no preprocessing of filling missing values is to be done

# In[4]:


data.isnull().sum()


# ### Checking the count of both the outcomes
# - As the probelm is of binary classification , we first checked the skewness of the data towards an outcome.
# - We can see that the zeroes count is much more than the ones count hence the tree will be biased

# In[5]:


data.groupby('left').count()


# ### Shuffling and spliting the data
# - As mentioned in the assignment, we will first shuffle the data and then split it into two splits
# - 80% Training Set
# - 20% Testing Set

# In[6]:


# train_data,test_data = train_test_split(data, test_size=0.2)

# msk = numpy.random.rand(len(data)) < 0.8

# train_data = data[msk]

# test_data = data[~msk]
train_data = data.sample(frac = 0.8, random_state = 200)
test_data = data.drop(train_data.index)
#train_data, test_data = numpy.split(data, [int(0.8 * len(data))])
train_data.head()
test_data.head()


# #### This function will return unique values of a particular column given as parameter

# In[7]:


def get_unique_values(data,col):
    return list(data[col].unique())


# In[8]:


get_unique_values(data,'salary')


#  #### This function will list the column headers for any dataframe.

# In[9]:


def get_attributes(data):
    return list(data)


# ## Entropy :
# >- Entropy is the term of imputiry of the system, that is how much variation is present in the outcome.
# >- The following function will take a dataframe and output label as an input and will calculate the entropy on the basis of frequency of output labels.
# 

# In[10]:


def entropy(data, labelcol):

    dic = dict(data[labelcol].value_counts())
    entropy = 0.0
    for label in dic.keys():
        entropy = entropy + (-(((1.0)*dic[label])/(len(data) + eps) * numpy.log2(((1.0)*dic[label])/(len(data) + eps )) ) ) 
        #print(entropy)
    return entropy
    


# In[11]:


entropy(train_data,'left')


# ## Information Gain
# >- information gain is reduction in the entropy if we use a feature ( attribute ) as an decision boundary.
# >- It calculates wieghted entropy for the unique values of the attribute and then subtracts it from the current entropy of the system.

# In[12]:


def infogain(data, attr,labelcol):
    uniq = get_unique_values(data,attr)
    wt_entropy = 0.0
    for vals in uniq:
        selected_data = data.loc[data[attr] == vals]
        wieght = (1.0*len(selected_data))/(len(data) + eps )
        wt_entropy += wieght*entropy(selected_data,labelcol)
        #print(wt_entropy)
    return entropy(data, labelcol) - wt_entropy


# In[13]:


infogain(train_data,'salary','left')


# #### Maximum Gain
# - Out of a list of attributes, we have to select that which attribute will be selected as a best decision boundary and for that we calculate inforamtion gain of eact feature and select the one whcih has max gain.

# In[14]:


def max_gain(data, remaining_attrs , labelcol):
    maxgain = 0.0
    splitattr = ""
    for attr in remaining_attrs:
        #print(attr)
        gain  = infogain(data,attr,labelcol)
        #print(gain)
        if  gain >= maxgain:
            maxgain = gain
            splitattr = attr

    if splitattr == "" :
        return 0,"negative"
    return maxgain,splitattr


# In[15]:


entropy(data,'left')


# In[16]:


attr = list(data)
attr.remove('left')
max_gain(data,attr,'left')


# ### In first part we have to only consider categorical features. 
# >- This function will seprate categorical and continous features.

# In[17]:


def iscontinous(data,attr):
    if attr == 'left':
        return True
    if attr in list(data._get_numeric_data()):
        if len(data[attr].unique()) <= 2 :
            return False
        return True
    return False


# In[18]:


categorical = [ x for x in attr if iscontinous(data,x) == False]
continous = [x for x in attr if iscontinous(data,x) == True]
print(categorical)
print(continous)


# - This function return the part of the dataframe where 'attr = value' that is column-attr has value-value

# In[19]:


def getcoldata(data,attr,value):
    return data.loc[data[attr] == value]


# In[20]:


getcoldata(data,'salary','low').head()


# ## Node of decision tree.
# >- I have build id3 type structure of the tree that is each node will have n-ary children depending upon the number of unique values of the feature selected on that node.
# >- Each node has a dictionary which will have key ( branch ) as its unique values and for each key there will be a node.
# >- I have taken following attribute for the node
# >>- attribute name selceted on that node
# >>- decision on that node
# >>- Dictionary of children
# >>- depth
# >>- positive count ( number of 1s )
# >>- negative count ( numbers of 0s )
# 

# In[21]:


class node:
    attr = ""
    decision = -1
    children = {}
    depth = 0
    positive_count = 0
    negative_count = 0

    
    def __init__(self,attr,decision,children,depth,positive_count,negative_count):
        self.attr = attr
        self.decision = decision
        self.children = children
        self.depth = depth
        self.positive_count = positive_count
        self.negative_count = negative_count

        


# ## Building tree recursively
# >- We first select the attribue with maximum gain , make it the root node and recurse the funciton for all its unique value.
# >- We stop at two conditions 
# >>1. If there are rows only of onr type of output
# >>2. If all the attributes are used in the path.
# >>> In that case we make decision on the probablity of outcome till that node.

# In[22]:


def building_tree(data ,attrs , depth ):
    #print(attrs)

    posnegcount = dict(data['left'].value_counts())
    poscount = 0
    negcount = 0
    
    if 1 in posnegcount.keys():
        poscount = posnegcount[1]
    if 0 in posnegcount.keys():
        negcount = posnegcount[0]
        
    
    gain , best_attr  = max_gain(data,attrs,'left')
    if(gain == 0):
       
        ans = 1
        if negcount > poscount:
            ans = 0
            
        return node('left',ans,{},depth,poscount,negcount)

    if  len(attrs) == 0 or poscount == 0 or negcount == 0:
        #print(str(list(data['left'])) + " list" +" gain :" + str(gain))
        ans = 1
        if negcount > poscount:
            ans = 0
            
        return node('left',ans,{},depth,poscount,negcount)
    else:
       
        uniqvals = get_unique_values(data,best_attr)
        root = node( best_attr , "" , {} , depth , poscount , negcount)
        for val in uniqvals:
            data2 = getcoldata(data,best_attr,val)
            newattr = deepcopy(attrs)
            newattr.remove(best_attr)
            global maxdepth
            maxdepth = max(depth,maxdepth)
            root.children[val] = building_tree(data2, newattr , depth+1)
        
        return root
            


# In[23]:


maxdepth = 0
tree = building_tree(train_data,categorical,0)
print("Root:",tree.attr)
print("depth: ",maxdepth)


# #### For my understanding, i printed preorder of the tree and counted the number of leaf nodes

# In[24]:


def printtree(root):
    if(len(root.children.keys()) == 0):
        #print("descion " + str(root.decision)) 
        global countdes
        countdes += 1
        return
    #print(root.attr)
    for x in list(root.children.keys())[::-1]:
        #print(x)
        printtree(root.children[x])


# In[25]:


countdes = 0
printtree(tree)
countdes


# ## Validation Function:
# > We traverse from root node for each row till the decison node, and then predict the same.
# >#### In case, if a path is not in the trained tree and it is in test set (that is we cant parse downwars stuck on that node, we give decsion on the basis of probablity of that node.

# In[26]:


def predict(tree, rows):
    while(len(tree.children.keys()) != 0):
        x = rows[tree.attr]
        try:
            tree = tree.children[x]
        except:
            if(tree.positive_count > tree.negative_count):
                tree.decision = 1
            else:
                tree.decision = 0
            break
    return tree.decision


# In[29]:


def predict2(tree, valid_data):
    rightcount = 0
    wrongcount = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    origin = deepcopy(tree)
    for index,rows in valid_data.iterrows():
        tree = origin
        decision = predict(tree,rows)
        #print(rows['left'],tree.decision )
        if(decision == rows['left']):
            rightcount+=1
            if(rows['left'] == 0):
                true_negative += 1
            if(rows['left'] == 1):
                true_positive += 1
        else:
            if(rows['left'] == 0):
                false_positive += 1
            if(rows['left'] == 1):
                false_negative += 1
            wrongcount+=1
    print("True negative : " , true_negative)
    print("True positive : ",true_positive)
    print("False Positive :",false_positive)
    print("False negative :",false_negative)
    print("Total right predicted: ", rightcount)
    print("Total wrong predicted: ", wrongcount)
    print("Accuracy: " , rightcount/(rightcount+wrongcount))
    print("Precision: ",true_positive/(true_positive+false_positive))
    print("Recall: ",true_positive/(true_positive + false_negative))
    print("F1 Score: ",(2.0)/((1/(true_positive/(true_positive + false_positive))) + (1/(true_positive/(true_positive+false_negative)))))


# In[30]:


predict2(tree,test_data)

