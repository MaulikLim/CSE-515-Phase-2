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
import numpy.linalg as la

parser = argparse.ArgumentParser(description="Task 9")

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
parser.add_argument(
    "-subid",
    type=str,
    required = True
)

args = parser.parse_args()

l_features = load_json(args.latent_path)
n = int(args.n)
m = int(args.m)
sub_ids = args.subid.split("-")
a = np.array(l_features[2])
labels = np.array(l_features[0])
rows,cols = a.shape
s = np.random.random((40,40))
topn = np.zeros((40,n))
for i in range(len(sub_ids)):
    sub_ids[i] = np.where(labels==sub_ids[i])[0][0] + 1
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
def generate_transitionMatrix(subject_subject_matrix) :
    subject_subject_matrix_transpose = subject_subject_matrix.transpose()
    answer_matrix = np.zeros((subject_subject_matrix_transpose.shape[0],subject_subject_matrix_transpose.shape[1]))
    for j in np.arange(subject_subject_matrix_transpose.shape[0]):
        columnSum = np.sum(np.absolute(subject_subject_matrix_transpose[j]))
        for i in np.arange(subject_subject_matrix_transpose.shape[1]):
            if columnSum==0:
                answer_matrix[j][i] = subject_subject_matrix_transpose[j][i]
                continue
            answer_matrix[j][i] = subject_subject_matrix_transpose[j][i]/columnSum
    return answer_matrix.transpose()
def generate_seedMatrix(subject_IDs,length):
    seedmatrix = np.zeros((length,1))
    for ids in subject_IDs:
        seedmatrix[ids-1]=1
    return seedmatrix
def pageRank(linkMatrix, d, seedmatrix) :
    n = linkMatrix.shape[0]
    #print("linkmatrix shape =",linkMatrix.shape)
    #M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100 * np.ones((n,1)) / n # Sets up this vector (n entries of 1/n × 100 each)
    #print("r shape = ",r.shape)
    last = r
    r = d*np.matmul(linkMatrix,r) + (1-d)/n * seedmatrix#M @ r
    #print("r shape after calculation= ",r.shape)
    while la.norm(last - r) > 0.01 :
        last = r
        r = d*np.matmul(linkMatrix,last) + (1-d)/n * seedmatrix
    return r
def task9(subject_subject_Matrix,subjectIds,m):
    transition_matrix = generate_transitionMatrix(subject_subject_Matrix)
    #transition_matrix = generate_transitionMatrix(40)
    seed_matrix = generate_seedMatrix(subjectIds,subject_subject_Matrix.shape[0])
    pageranks = pageRank(transition_matrix,0.85,seed_matrix)
    subject_rank = dict()
    i=0
    pageranks.reshape((pageranks.shape[0],))
    #print(pagerank.shape)
    for rank in pageranks:
        subject_rank[labels[i]] = rank[0]
        i+=1
    reverse_subject_rank = sorted(subject_rank.items(),key = lambda x : x[1],reverse = True)
    topm = m
    for i in dict(reverse_subject_rank).keys():
        if(topm>0):
            print(i)
            topm=topm-1
        
        #print(i)

    # for i in np.arange(m):
    #     row = reverse_subject_rank[i]
    #     print(row[0],row[1])
task9(temp, sub_ids, m)
