#!/usr/bin/env python
# coding: utf-8

# ## Part -1

# ### importing modules

# In[7]:


import pandas as pd
import random
import numpy as  np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import operator
from copy import deepcopy
from collections import Counter
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering


# ### dropping output column

# In[8]:


original_data = pd.read_csv('data.csv')
total_y_values = original_data['xAttack']
original_data = original_data.drop('xAttack',axis = 1)


# ### Normalizing data

# In[9]:


data = pd.read_csv('data.csv')
total_y_values = data['xAttack']
data = data.drop('xAttack',axis = 1)

data=data.astype('float128')
data_mean = deepcopy(data.mean())
data_std = deepcopy(data.std())
data =(data-data_mean)*np.float128(1.0)/data_std
print(data.head())


# In[10]:


data.iloc[:1,:]


# ### Neural Net code

# In[11]:


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))


# In[11]:


def sigmoid_der(z):
    return np.multiply(sigmoid(z), (1.0 - sigmoid(z)))


# In[12]:


def relu(z):
    return np.maximum(z, 0)


# In[13]:


def relu_der(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# In[14]:


def tanh(x):
    return np.tanh(x)


# In[15]:


def tanh_der(x):
    return 1.0 - (np.power(tanh(x), 2))


# In[16]:


def softmax(Z):
    expA = np.exp(Z - Z.max(1).reshape(Z.shape[0], 1))
    esum = expA / np.sum(expA, 1).reshape(Z.shape[0], 1)
    return esum

def linear(z):
    return z

def linear_der(z):
    return 1


# In[17]:


def softmax_der(x):
    y = deepcopy(x)
    edeno = np.power(np.sum(np.exp(x)), 2)
    enumo = np.sum(np.exp(x))
    for ele in y:
        for k in range(len(ele)):
            tempy = y[0][k]
            y[0][k] = (np.exp(tempy) * (enumo - np.exp(tempy)) / edeno)
    return y


# ### Layer class
# >- layer class will hold all the information regarding each layer
# >- each layer will have many attributes such as
# >>- number of nodes in each layer
# >>- activation function
# >>- derivatives regarding gradient decent

# In[18]:


class layer:
    def __init__(self, layerno, num_of_nodes, activation_function, isinput,
                 isoutput, ishidden):
        self.layerno = layerno
        self.num_of_nodes = num_of_nodes
        self.activation_function = activation_function
        self.is_input_layer = isinput
        self.is_ouput_layer = isoutput
        self.is_hidden_layer = ishidden
        self.input = None
        self.output = None
        self.error_cost = np.float128(1.0)
        self.derivative_op = np.float128(1.0)
        self.derivative_act = np.float128(1.0)
        self.derivative_wt = np.float128(1.0)
        self.k_product = np.float128(1.0)


# ### Function that generate random weights of shape of ( layer n-1 , layer n )

# In[19]:


def getrandom_wts(numberofnodes_prev, numberofnodes_next):
    return (0.01 * (np.random.randn(numberofnodes_prev, numberofnodes_next)))


# ### Common activation fucntion that will be called

# In[20]:


def activation(z, function_name):
    if function_name == 'sigmoid':
        return sigmoid(z)
    if function_name == 'tanh':
        return tanh(z)
    if function_name == 'relu':
        return relu(z)
    if function_name == "softmax":
        return softmax(z)
    if function_name == "linear":
        return linear(z)


# In[21]:


def activation_der(z, function_name):
    if function_name == 'sigmoid':
        return sigmoid_der(z)
    if function_name == 'tanh':
        return tanh_der(z)
    if function_name == 'relu':
        return relu_der(z)
    if function_name == "softmax":
        return softmax_der(z)
    if function_name == "linear":
        return linear_der(z)


# ### Class neural net
# >- It holds the structure of the neural net
# >- It has following methods
# >- 1. Initializer :
# >>- it intializes the hyper parameters like learning rate, number of epochs and batch size.
# >- 2. Add layer:
# >>- It adds a layer to neural net, we can define activation function, number of nodes in each layer.
# >- 3. Forward propogation
# >- 4. Backward propogation
# >- 5. Predict
# >- 6. Fit
# >>- Train the neural network using forward and back ward propogation

# In[123]:


class neural_net:
    layerno = 0

    def __init__(self, numboflayer, learning_rate, epochs, batch_size):
        self.numboflayer = numboflayer
        self.layers = []
        self.weights = []
        self.bias = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_matrix = None
        self.batch_size = batch_size
        self.error_each_epoch = []
        self.acc_each_epoch = []
        neural_net.layerno = 0
        self.compressed_output = None
        self.bottleneck_layer_no = (numboflayer - 1)//2

    def predict(self, y, actual_y):
        rcount = 0
        wcount = 0
        for ind, i in enumerate(y):
            self.forward_propogation(np.array(i).reshape(1, len(i)))
            list_of_op = list(np.ndarray.flatten(self.layers[-1].output))
            index = list_of_op.index(max(list_of_op))
            if (actual_y[ind][index] == 1):
                rcount += 1
            else:
                wcount += 1
        print("right count: ", rcount)
        print("wrong count: ", wcount)
        print("Accuracy", rcount / (rcount + wcount))
        self.acc_each_epoch.append(rcount / (rcount + wcount))
        print()

    def predict_2(self, y):
        rcount = 0
        wcount = 0
        f = open("2018201004_prediction.csv", "w+")
        for ind, i in enumerate(y):
            self.forward_propogation(np.array(i).reshape(1, len(i)))
            list_of_op = list(np.ndarray.flatten(self.layers[-1].output))
            index = list_of_op.index(max(list_of_op))
            f.write(str(int(index)))
            if ind != (len(y) - 1):
                f.write("\n")

    def test_data(self, test_inp, test_op):
        self.test_inp = test_inp
        self.test_op = test_op

    def fit(self, x, y):
        for epoch in range(self.epochs):
            self.forward_propogation(x)
            self.backward_propogation(y)
            if epoch%25 == 0:
                print("Epoch",epoch)
                print("error",self.layers[-1].error)

#             if(error_batch/len(x) < 15 and not flag_of_learningrate):
#                 self.learning_rate = self.learning_rate/10
#                 flag_of_learningrate = 1
#            print(self.layers[-1].output)
#            self.predict(self.test_inp, self.test_op)
#             for layer in self.layers:
#                 if layer.layerno != 0:
#                     print("weights",self.weights[layer.layerno-1])
#                 print("layerno",layer.layerno)
#                 print("input",layer.input)
#                 print("output",layer.output)
#                 print("serivative1",layer.derivative_op)
#                 print("derivative2",layer.derivative_act)
#                 print("kproduct",layer.k_product)
#                 print("derivative3",layer.derivative_wt)

    def add_layer(self, num_of_nodes, activation_function):
        if neural_net.layerno == 0:
            temp_layer = layer(neural_net.layerno, num_of_nodes,
                               activation_function, True, False, False)
        elif neural_net.layerno == self.numboflayer - 1:
            temp_layer = layer(neural_net.layerno, num_of_nodes,
                               activation_function, False, True, False)
        else:
            temp_layer = layer(neural_net.layerno, num_of_nodes,
                               activation_function, False, False, True)

        neural_net.layerno += 1

        self.layers.append(temp_layer)

        if (temp_layer.is_input_layer == False):
            #self.weights.append(random_wts[temp_layer.layerno-1])
            #print(temp_layer.layerno)
            self.weights.append(
                getrandom_wts(self.layers[temp_layer.layerno - 1].num_of_nodes,
                              temp_layer.num_of_nodes))
            self.bias.append(np.random.randn(1, temp_layer.num_of_nodes))

        #self.weights = np.array(self.weights)
    def get_compressed_datamatrix(self,x):
        output_rows = []
        for rowindex in range(len(x)):
            self.forward_propogation(
                    np.array(x[rowindex]).reshape(1, len(x[rowindex])))
            compressed_row = np.ndarray.flatten(self.layers[self.bottleneck_layer_no].output)
            output_rows.append(compressed_row)
        return np.stack(output_rows)

    def forward_propogation(self, x):

        for layer in self.layers:
            # print(layer.layerno)
            if layer.layerno == 0:
                layer.input = x
                if layer.activation_function:
                    layer.output = activation(layer.input,
                                              layer.activation_function)
                else:
                    layer.output = x
            else:
                layer.input = np.dot(
                    self.layers[layer.layerno - 1].output, self.weights[
                        layer.layerno - 1]) + self.bias[layer.layerno - 1]
                layer.output = activation(layer.input,
                                          layer.activation_function)
#                 if (layer.layerno + 1) < len(self.layers):
#                     self.layers[layer.layerno + 1].input = layer.output
# print(layer.input)
#  print(layer.output)

    def error(self, predicted, actual):
        return np.mean(np.power((np.array(predicted) - np.array(actual)), 2))

    def error_der(self, predicted, actual):
        return 2 * (predicted - actual)

    def error2(self, predicted, actual):
        epsilon = 1e-12
        predictions = np.clip(predicted, epsilon, 1. - epsilon)
        N = predictions.shape[1]
        ce = -np.sum(actual * np.log(predictions + 1e-9)) / N
        return ce

    def error2_der(self, predicted, actual):
        return -1 * ((actual * (1 / predicted)) + (1 - actual) * (
            (1 / (1 - predicted))))

    def weights_behind(self, layer_):
        return self.weights[layer_.layerno - 1]

    def weights_ahead(self, layer_):
        return self.weights[layer_.layerno]

    def layer_behind(self, layer_):
        return self.layers[layer_.layerno - 1]

    def layer_ahead(self, layer_):
        return self.layers[layer_.layerno + 1]

    def backward_propogation(self, y):

        stored = np.float128(1.0)

        for curr_layer in self.layers[::-1]:
            if curr_layer.layerno == 0:
                continue
            if curr_layer.layerno == len(self.layers) - 1:

                stored = np.dot(self.error(curr_layer.output, y), stored)
                curr_layer.error = stored
                # print("error",curr_layer.error)

                curr_layer.derivative_op = self.error_der(curr_layer.output, y)
                #  print(curr_layer.derivative_op,curr_layer.derivative_op.shape)

                curr_layer.derivative_act = activation_der(
                    curr_layer.input, curr_layer.activation_function)
                #  print(curr_layer.derivative_act,curr_layer.derivative_act.shape)

                curr_layer.k_product = curr_layer.derivative_op * curr_layer.derivative_act
                # print(curr_layer.k_product,curr_layer.k_product.shape)

                #curr_layer.k_product = curr_layer.k_product.reshape(1,curr_layer.derivative_act.shape[0])
                curr_layer.derivative_wt = np.dot(
                    (self.layers[curr_layer.layerno - 1].output).T,
                    curr_layer.k_product)

                #self.weights[curr_layer.layerno - 1] -= self.learning_rate * curr_layer.derivative_wt

            else:
                #   print(curr_layer.layerno)
                curr_layer.error = None
                curr_layer.derivative_act = activation_der(
                    curr_layer.input, curr_layer.activation_function)
                curr_layer.derivative_op = np.dot(
                    self.layers[curr_layer.layerno + 1].k_product,
                    (self.weights_ahead(curr_layer)).T)
                curr_layer.k_product = curr_layer.derivative_op * curr_layer.derivative_act
                #if curr_layer.layerno != 1:
                curr_layer.derivative_wt = np.dot(
                    (self.layer_behind(curr_layer).output).T,
                    curr_layer.k_product)


#                 else:
#                     curr_layer.derivative_wt = np.dot( (self.layer_behind(curr_layer).input).T , curr_layer.k_product)

        for w_no, weight in enumerate(self.weights):
            self.weights[w_no] -= (self.learning_rate/len(y)) * (
                self.layers[w_no + 1].derivative_wt)
            self.bias[w_no] -= (self.learning_rate/len(y)) * (
                np.sum(self.layers[w_no + 1].k_product , axis = 0))


# ### A. Linear 3 layer autoencoder

# In[12]:


net = neural_net(3,0.01,700,100)
net.add_layer(29,"linear")
net.add_layer(14,"linear")
net.add_layer(29,"linear")

data_np_version = np.array(data)

net.fit(data_np_version,data_np_version)

compressed_datamatrix_1 = net.layers[net.bottleneck_layer_no].output

compressed_datamatrix_1.shape


# In[89]:


output_label = kmeans_final(5,compressed_datamatrix_1)
print()
print("Purity of k means: ")
print()
p1 = calculate_purity(output_label)


# In[93]:


output_label = gmm_implementation(compressed_datamatrix_1)
print()
print("Purity of GMM : ")
print()
p2 = calculate_purity(output_label)


# In[94]:


output_label = hmm_implementation(compressed_datamatrix_1)
print()
print("Purity of hiearichal clustering: ")
print()
p3 = calculate_purity(output_label)


# In[95]:


pie_chart(p1,p2,p3)


# ### Non - linear 3-layer auto encoder

# In[96]:


net = neural_net(3,0.1,700,100)
net.add_layer(29,"linear")
net.add_layer(14,"sigmoid")
net.add_layer(29,"linear")

data_np_version = np.array(data)


net.fit(data_np_version,data_np_version)



compressed_datamatrix_2 = net.layers[net.bottleneck_layer_no].output


# In[120]:


output_label = kmeans_final(5,compressed_datamatrix_2)
print()
print("Purity of k means: ")
print()
p1 = calculate_purity(output_label)


# In[121]:


output_label = gmm_implementation(compressed_datamatrix_2)
print()
print("Purity of GMM : ")
print()
p2 = calculate_purity(output_label)
output_label = hmm_implementation(compressed_datamatrix_2)
print()
print("Purity of hiearichal clustering: ")
print()
p3 = calculate_purity(output_label)
pie_chart(p1,p2,p3)


# ### B. Non-linear 5 layer autoencoder

# >- relu

# In[122]:


net = neural_net(5,0.1,700,100)
net.add_layer(29,"linear")
net.add_layer(21,"sigmoid")
net.add_layer(14,"relu")
net.add_layer(21,"sigmoid")
net.add_layer(29,"linear")


# In[123]:


net.fit(data_np_version,data_np_version)


# In[125]:


compressed_datamatrix_3 = net.layers[net.bottleneck_layer_no].output


# In[211]:





# In[174]:


output_label = kmeans_final(5,compressed_datamatrix_3)
print()
print("Purity of k means: ")
print()
p1 = calculate_purity(output_label)
print()
output_label = gmm_implementation(compressed_datamatrix_3)
print()
print("Purity of GMM : ")
print()
p2 = calculate_purity(output_label)
output_label = hmm_implementation(compressed_datamatrix_3)
print()
print("Purity of hiearichal clustering: ")
print()
p3 = calculate_purity(output_label)
pie_chart(p1,p2,p3)


# >- Relu-sigmoid=relu

# ### Non - linear 5 - layer auto encoder

# In[177]:


net = neural_net(5,0.1,700,100)
net.add_layer(29,"linear")
net.add_layer(21,"sigmoid")
net.add_layer(14,"linear")
net.add_layer(21,"sigmoid")
net.add_layer(29,"linear")

net.fit(data_np_version,data_np_version)


# In[178]:


compressed_datamatrix_4 = net.layers[net.bottleneck_layer_no].output


# In[194]:


output_label = kmeans_final(5,compressed_datamatrix_4)
print()
print("Purity of k means: ")
print()
p1 = calculate_purity(output_label)
print()
output_label = gmm_implementation(compressed_datamatrix_4)
print()
print("Purity of GMM : ")
print()
p2 = calculate_purity(output_label)
output_label = hmm_implementation(compressed_datamatrix_4)
print()
print("Purity of hiearichal clustering: ")
print()
p3 = calculate_purity(output_label)
pie_chart(p1,p2,p3)


# ### Non linear 7 layer network

# In[195]:


net = neural_net(7,0.5,150,100)
net.add_layer(29,"linear")
net.add_layer(21,"sigmoid")
net.add_layer(17,"sigmoid")
net.add_layer(14,"sigmoid")
net.add_layer(17,"sigmoid")
net.add_layer(21,"sigmoid")
net.add_layer(29,"linear")

net.fit(data_np_version,data_np_version)
compressed_datamatrix_5 = net.layers[net.bottleneck_layer_no].output


# In[208]:


output_label = kmeans_final(5,compressed_datamatrix_4)
print()
print("Purity of k means: ")
print()
p1 = calculate_purity(output_label)
print()
output_label = gmm_implementation(compressed_datamatrix_4)
print()
print("Purity of GMM : ")
print()
p2 = calculate_purity(output_label)
output_label = hmm_implementation(compressed_datamatrix_4)
print()
print("Purity of hiearichal clustering: ")
print()
p3 = calculate_purity(output_label)
pie_chart(p1,p2,p3)


# ### Code for K - Means clustering

# In[137]:



def ecludian(row1,row2):
    dist = 0.0
    dist += math.sqrt(sum((np.array(row1)-np.array(row2))**2))
    return dist


# In[79]:


def assigncluster(centroid , row):
    mintill = 9999999999
    clust = -1
    for i in centroid:
        dist  = ecludian(row,i)
        if(dist < mintill):
            mintill = dist
            clust = i
    return clust


# In[80]:


def recalculate_centroid(clusters,data):
    newcentroid = []
    for i in clusters.keys():
        sumc = [0]*len(clusters[i][0])
        for j in range(len(clusters[i])):
            #print(clusters[i],"clusteri")
            #print(clusters[i][j],"what is this")
            sumc = sumc + np.array(clusters[i][j])  #data.loc[index.get_loc(clusters[i][j] )]
            #print(sumc,"sums")                       
        newcentroid.append(tuple(sumc/len(clusters[i])))
    return newcentroid


# In[81]:


def getrandomcentorid(k,data):
    centroids = []
    for i in range(k):
        rand = random.randint(0,len(data)-1)
        randval = tuple(data.loc[rand].values)
        while randval in centroids:
            randval = tuple(data.loc[random.randint(0,len(data)-1)])
        centroids.append(randval)
    return centroids


# In[82]:


def buildcluster(k,data,centroids):
    clu = {}
    for c in centroids:
        clu[c] = []
    for i,row in data.iterrows():
        rown = tuple(row)
        #print(assigncluster(centroids,rown))
        clu[assigncluster(centroids,rown)].append(rown)
        #print(rown)
    return clu


# In[83]:


def converge(cone,ctwo,tolerance,k):
    countofpos = 0
    for i in range(k):
        dist = ecludian(cone[i],ctwo[i])
        if(dist <= tolerance):
            countofpos += 1
    if(countofpos == k):
        return True
    else:
        return False


# In[ ]:





# In[84]:


def kmeans(data,k):
    centroid = getrandomcentorid(k,data)
    c1 = deepcopy(centroid)
    
    #print(c1)
    #print(len(c1))
    #print(assigncluster(c1,c1[0]),"assigned")
    #print(c1)
    clusters = buildcluster(k,data,c1)
    #print(clusters)
    #print(len(clusters.keys()),"keys")
    newcentroid = recalculate_centroid(clusters,data)
    iterations = 0
    while not converge(c1,newcentroid,0,k) :
        print(iterations)
        c1 = newcentroid
        clusters = buildcluster(k,data,c1)
        newcentroid = recalculate_centroid(clusters,data)
        iterations += 1
    return clusters
    print("final clusters",clusters)


# In[85]:


#data_for_kmeans = pd.DataFrame(pro_x)
#data_for_kmeans = (data_for_kmeans - data_for_kmeans.mean())/data_for_kmeans.std()
# print(data_for_kmeans)
# data_for_kmeans.describe()
#data_for_kmeans.columns = list(data)
# data_for_kmeans = data_for_kmeans
# clusters = kmeans(data_for_kmeans,5)

# clusterone = clusters[list(clusters.keys())[4]]
# results = []
# index = -1
# for i in clusterone:
#     for no,j in original_data.iterrows():
#         if tuple(j) == i:
#             index = no
#             break
#     results.append(total_y_values.iloc[index])


# In[86]:


def getdistances(data,centers,i):
    return  np.linalg.norm(data - centers[i], axis=1)


# In[87]:


def geterror(centernew,centerold):
    return np.linalg.norm(centernew - centerold)


# In[88]:


def getnewcenters(k,data,centers_new,clusters):
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    return centers_new


# In[116]:





# In[117]:


# In[159]:


def kmeans_final(k,pro_x):
    n = pro_x.shape[0]
    data = pro_x
    #print(pro_x.shape[1])


    centers = np.random.randn(k,pro_x.shape[1])
    #print(centers)
    centers_old = np.zeros(centers.shape) 
    #print(centers_old.shape)
    centers_new = deepcopy(centers) 

    #print(data.shape)
    clusters = np.zeros(n)
    #print()
    distances = np.zeros((n,k))
    #print(distances.shape)

    error = geterror(centers_new,centers_old)

    iterations = 0 
    while error >= 0.00: #and iterations <= 1000:

        for i in range(k):
            distances[:,i] = getdistances(data,centers,i)

        clusters = np.argmin(distances, axis = 1)

        centers_old = deepcopy(centers_new)

        centers_new = getnewcenters(k,data,centers_new,clusters)


        error = geterror(centers_new,centers_old)
        iterations += 1

    purity = []
    output_labels = []

    for i in range(k):
        output_labels.append([])

    total_for_k_means = np.array(total_y_values)
    for i in range(n):
        output_labels[clusters[i]].append(total_y_values[i])
    return output_labels


# In[170]:


def calculate_purity(output_labels):
    purity = []
    for i in range(5):
        if(len(output_label[i]) != 0 ):
            max_label = max(set(output_labels[i]), key=output_labels[i].count)
            #print(max_label)
            dictio = Counter(output_labels[i])
            purity.append(dictio[max_label]/len(output_labels[i]))
            
            print("cluster : ",i,", Maximum label present : ",max_label,", No of times: ",dictio[max_label],", Purity :",purity[i])
        if(len(output_label[i]) == 0 ):
            purity.append(0)
    sumavg = 0
    cnt = 0
    for i in purity :
        if(i != 0 ):
            sumavg += i
            cnt += 1
    avg_purity = sumavg/cnt
    print("Avg purity : ", avg_purity)
    return purity
    


# In[301]:


# for i in output_labels:


# ### Code for GMM

# In[3]:


def gmm_implementation(compressed_datamatrix_1):
    clf = mixture.GaussianMixture(n_components=5)
    labels = clf.fit_predict(compressed_datamatrix_1)
    k = 5
    n = len(compressed_datamatrix_1)
    print(Counter(labels))
    output_labels3 = []
    for i in range(k):
        output_labels3.append([])

    total_for_k_means = np.array(total_y_values)
    for i in range(n):
        output_labels3[labels[i]].append(total_y_values[i])
    return output_labels3


# In[302]:



clf = mixture.GaussianMixture(n_components=5)
labels = clf.fit_predict(compressed_datamatrix_1)
k = 5
n = len(compressed_datamatrix)



print(Counter(labels))


# In[95]:


output_labels3 = []

for i in range(k):
    output_labels3.append([])

total_for_k_means = np.array(total_y_values)
for i in range(n):
    output_labels3[labels[i]].append(total_y_values[i])


# In[96]:


for i in output_labels3:
    print(len(i))
purity3 = []
for i in range(k):
    max_label = max(set(output_labels3[i]), key=output_labels3[i].count)
    dictio = Counter(output_labels3[i])
    purity3.append(dictio[max_label]/len(output_labels3[i]))
    print("cluster : ",i,", Maximum label present : ",max_label,", No of times: ",dictio[max_label],", Purity :",purity3[i])

print("Avg purity : ", sum(purity3)/len(purity3))


# ### Code for hiraerichal clustering

# In[5]:


def hmm_implementation(compressed_datamatrix_1):
    cluster_hir = AgglomerativeClustering(n_clusters=5, linkage='single')
    labels4 = cluster_hir.fit_predict(compressed_datamatrix_1)
    k = 5
    n = len(compressed_datamatrix_1)


    print(Counter(labels4))
    output_labels4 = []

    for i in range(k):
        output_labels4.append([])

    total_for_k_means = np.array(total_y_values)
    for i in range(n):
        output_labels4[labels4[i]].append(total_y_values[i])
    return output_labels4


# In[303]:


from sklearn.cluster import AgglomerativeClustering

cluster_hir = AgglomerativeClustering(n_clusters=5,  linkage='single')  
labels4 = cluster_hir.fit_predict(compressed_datamatrix_1)
k = 5
n = len(compressed_datamatrix)

# In[98]:


print(Counter(labels4))


# In[99]:


output_labels4 = []

for i in range(k):
    output_labels4.append([])

total_for_k_means = np.array(total_y_values)
for i in range(n):
    output_labels4[labels4[i]].append(total_y_values[i])


# In[100]:


for i in output_labels4:
    print(len(i))
purity4 = []
for i in range(k):
    max_label = max(set(output_labels4[i]), key=output_labels4[i].count)
    dictio = Counter(output_labels4[i])
    purity4.append(dictio[max_label]/len(output_labels4[i]))
    print("cluster : ",i,", Maximum label present : ",max_label,", No of times: ",dictio[max_label],", Purity :",purity4[i])

print("Avg purity : ", sum(purity4)/len(purity4))


# ### Code for pie chart generation

# In[6]:


def pie_chart(purity,purity3,purity4):
    avg_purities = [sum(purity)/len(purity) , sum(purity3)/len(purity3) , sum(purity4)/len(purity4)]
    labels = ["K-NN","GMM","Hierarchical"]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    plt.figure(figsize=(10,12))
    explode = (0, 0, 0.1)
    plt.pie(avg_purities , explode = explode, labels=labels, colors=colors ,
    autopct='%1.1f%%', shadow=True, startangle=140 )

    plt.axis('equal')
    plt.show()


# ### Testing with keras model of 7 layer model

# In[306]:


model = Sequential()
model.add(Dense(21,input_dim=29,activation="sigmoid"))
model.add(Dense(17,input_dim=21,activation="sigmoid"))
model.add(Dense(14,input_dim=17,activation="sigmoid"))
model.add(Dense(17,input_dim=14,activation="sigmoid"))
model.add(Dense(21,input_dim=17,activation="sigmoid"))
model.add(Dense(29,input_dim=21,activation=None))


model.compile(optimizer='sgd',loss='mean_squared_error', metrics=['accuracy'])


# In[308]:


model.fit(data_np_version,data_np_version,epochs=20)


# In[79]:


predictions = model.predict(data_np_version[:,:])


# In[80]:


l = np.square(predictions[:,:] - data_np_version[:,:])


# In[81]:


b = np.sum(l,axis=1)


# In[82]:


c = np.sum(b,axis = 0)


# In[84]:


c/(24998*29)


# In[85]:


d = np.mean(np.square(predictions - data_np_version),axis=-1)


# In[86]:


np.mean(d)


# In[23]:



data_np_version[:1,:]

