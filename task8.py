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
    "-sp",
    "--sub_path",
    type=str,
    required=True,
)

args = parser.parse_args()

sub_sub_mat = load_json(args.sub_path)
n = int(args.n)
m = int(args.m)
a = np.array(sub_sub_mat[1])
labels = np.array(sub_sub_mat[0])
rows, cols = a.shape
s = np.random.random(a.shape)


def generate_topn(n):
    topn = np.zeros((a.shape[0], n))

    for i in range(rows):
        temp = []
        temp = np.argpartition(a[i], -n)[-n:]  # gets indices of top n cells of each row
        for j in range(n):
            topn[i][j] = temp[j]

    temp = np.zeros(a.shape)
    for i in range(len(topn)):
        for j in range(len(topn[i])):
            temp[i][int(topn[i][j])] = a[i][int(topn[i][j])]
    return temp


def generate_sum(temp):
    sumweight = np.zeros(a.shape[0])

    for i in range(rows):
        for j in range(cols):
            sumweight[i] += temp[j][i]
    return sumweight


def task8(temp, n, m):
    diff = 100000
    c = 0.85
    temp = generate_topn(n)
    sumweight = generate_sum(temp)
    while diff > 0.0000001:
        diff = 0
        for i in range(rows):
            for j in range(cols):
                if i == j:
                    s[i][j] = 1
                    continue
                sum = 0
                for k in range(rows):
                    # print(sumweight[i])
                    if temp[k][i] == 0:
                        continue
                    sum += (temp[k][i] / sumweight[i]) * (s[k][j]) * (1 - math.exp(-temp[k][i]))
                diff += abs(s[i][j] - c * sum)
                s[i][j] = c * sum

    topK = np.zeros((rows, 1))
    for i in range(rows):
        topK[i] = np.sum(s[:][i]) / rows
    indices = np.flip(np.argsort(topK, axis=0))
    # print(labels)
    # print(indices[:m])
    for ind in indices[:m]:
        print(int(labels[ind][0]))


task8(sub_sub_mat, n, m)
