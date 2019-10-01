import numpy as np
import pandas as pd
from pprint import pprint
from numpy import linalg

def find_best_K(S):
    for i in range(len(S)):
        p = (np.sum(S[:i])/np.sum(S))
        if p > 0.9:
            return i    

def PCA(frame):
    frame = (frame - frame.mean())/frame.std()
    
    X = frame.values
    covMat = np.cov(X.T)
    U, S, V = linalg.svd(covMat)
    K = find_best_K(S)
    print('K = ',K)
    Ureduced = U[:,:K]
    Z = np.matmul(X ,Ureduced)
    return Z

def main():
    frame = pd.read_csv('intrusion detection/data.csv').drop('xAttack')
    Z = PCA(frame)
    print("Size of data frame after applying PCA = ",Z.shape)
