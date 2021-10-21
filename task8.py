from numpy.lib import math
from featureGenerator import save_features_to_json
from featureLoader import load_json
import imageLoader
import modelFactory
import argparse
import json
import os
import numpy as np
import datetime
from tech.SVD import SVD
from utilities import intersection_similarity_between_features, print_semantics_type

parser = argparse.ArgumentParser(description="Task 8")

parser.add_argument(
    "-n",
    "--n",
    type=int,
    required=True,
)

parser.add_argument(
    "-m",
    "--m",
    type=int,
    required=True,
)
parser.add_argument(
    "-lp",
    "--latent_path",
    type=str,
    required=True,
)

args = parser.parse_args()

l_features = load_json(args.latent_path)
n = int(args.n)
m = int(args.m)
a = np.array(l_features[2])
labels = np.array(l_features[0])
rows,cols = a.shape
s = np.random.random((40,40))

topn = np.zeros((40,n))

for i in range(rows):
    temp = []
    temp = np.argpartition(a[i], -n)[-n:] #gets indices of top n cells of each row
    for j in range(n):
        topn[i][j] = temp[j]
#     print(topn[i])
    topn[i].sort()
temp = np.zeros((40,40))
for i in range(len(topn)):
    for j in range(len(topn[i])):
        temp[i][int(topn[i][j])] = a[i][int(topn[i][j])]

diff = 100000
c=1
sumweight=np.zeros((40,1))

for i in range(rows):
    for j in range(cols):
        sumweight[i]+=temp[i][j]
# print(sumweight)
    
        
while(diff<50):
    diff = 0
    for i in range(rows):
        for j in range(cols):
            if i==j:
                s[i][j]=1
                continue
            sum=0
            for k in range(topn[i]):
                sum+=(a[i][k]/sumweight[i])*(s[k][j])*(1-math.exp(a[j][k]))
            diff = math.abs(s[i][j]-c*sum)
            s[i][j] = sum
topK = np.zeros((rows,1))
for i in range(rows):
    topK[i] = np.sum(s[:][i])/rows
indices = np.flip(np.argsort(topK, axis = 0))
for ind in indices[:m]:
    print(labels[ind])
