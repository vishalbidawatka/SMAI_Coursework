#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import numpy as  np
import math
from copy import deepcopy
from sklearn.model_selection import KFold
import operator
import sys



def general(testfile):
    data = pd.read_csv('LoanDataset/data.csv',names = ["ID", "Age", "experience", "Annual Income","ZIPCode","Family size","Avgerage spending per month","Education Level","Mortgage Value of house","Output","securities account","certificate","internet banking","credit card"])
    valid_data = pd.read_csv(testfile,names = ["ID", "Age", "experience", "Annual Income","ZIPCode","Family size","Avgerage spending per month","Education Level","Mortgage Value of house","securities account","certificate","internet banking","credit card"] )





    data = data[1:]



    data.drop('ID',axis=1,inplace = True)
    data.drop('ZIPCode',axis=1,inplace = True)
    valid_data.drop('ID',axis=1,inplace = True)
    valid_data.drop('ZIPCode',axis=1,inplace = True)




    attributes = list(data)


    data[data < 0] = data.loc[data['experience'] > 0]['experience'].mean()
    valid_data[data < 0] = valid_data.loc[valid_data['experience'] > 0]['experience'].mean()


    train_data = data



    numerical_attr = ['Age',
    'experience',
    'Annual Income',
    'Family size',
    'Avgerage spending per month','Mortgage Value of house']




    categorical_attr = list(set(attributes) - set(numerical_attr))
    categorical_attr.remove('Output')



    totalyncounts = dict(train_data['Output'].value_counts())



    totalncount = totalyncounts[0.0]
    totalycount = totalyncounts[1.0]
    priory = totalycount*1.0/len(train_data)
    priorn = totalncount*1.0/len(train_data)




    def zeroonecount(df):
        l = [0,0]
        x = df['Output'].value_counts()
        for i,cnt in enumerate(x.keys()):
            l[not cnt] = x[cnt]
        return l



    def meanandsd(df,output,attr):
        tempdf = df.loc[df['Output'] == output]
        return [tempdf[attr].mean(),tempdf[attr].std()]






    categorical_dict = {}
    numerical_dict = {}
    for category in categorical_attr:

        for uniq in train_data[category].unique():
            try:
                categorical_dict[category][uniq] =  zeroonecount(train_data.loc[train_data[category] == uniq])
            except:
                categorical_dict[category] = { uniq : zeroonecount(train_data.loc[train_data[category] == uniq])}

    for attr in numerical_attr:
        numerical_dict[attr] = {'yes':meanandsd(train_data,1.0,attr) , 'no':meanandsd(train_data,0.0,attr)}
            



    def guassiandist(attr,df,x,output):
        mean = numerical_dict[attr][output][0]
        std = numerical_dict[attr][output][1]
        exp = math.exp(-(math.pow(x-mean,2))/(2*math.pow(std,2)))
        ans = (1.0/math.sqrt(2*math.pi)*std)*exp
        return ans



    wcount = 0
    rcount  = 0
    tp =0
    tn = 0
    fp = 0
    fn = 0
    for rowno,row in valid_data.iterrows():
        yesprob = 1.0
        noprob = 1.0
        
        for catattr in categorical_attr:
            yesprob *= max(((1.0*categorical_dict[catattr][row[catattr]][0])/totalycount),10**-9)
            noprob *= max(((1.0*categorical_dict[catattr][row[catattr]][1])/totalncount),10**-9)
        for numattr in numerical_attr:
            yesprob *= max(guassiandist(numattr,train_data,row[numattr],'yes'),10**-9)
            noprob *= max(guassiandist(numattr,train_data,row[numattr],'no'),10**-9)
        yesprob *= priory
        noprob *= priorn
 
        if yesprob >= noprob:
            decision = 1
        else:
            decision = 0

        print(decision)




if __name__ == "__main__":
   general(sys.argv[1])






