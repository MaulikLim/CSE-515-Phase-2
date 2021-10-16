from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import os
import numpy as np
import datetime

from tech.KMeans import KMeans
from tech.SVD import SVD
from tech.LDA import LDA
from utilities import print_semantics_type

parser = argparse.ArgumentParser(description="Task 3")
parser.add_argument(
    "-fp",
    "--folder_path",
    type=str,
    required=True,
)
parser.add_argument(
    "-f",
    "--feature_model",
    type=str,
    required=True,
)

parser.add_argument(
    "-k",
    "--k",
    type=int,
    required=True,
)
parser.add_argument(
    "-t",
    "--tech",
    type=str,
    required=True,
)
args = parser.parse_args()


def create_type_type(features, labels):
    res = []
    labs = []
    res_labs = []
    for i in range(len(labels)):
        lab = labels[i].split("-")[1]
        if lab not in labs:
            labs.append(lab)
            res_labs.append(labels[i])
            res.append(features[i])
        else:
            ind = labs.index(lab)
            res[ind] = np.mean(np.array([res[ind], features[i]]), axis=0)
    res = np.array(res)
    print(res.shape, "res")
    ans = np.matmul(res, res.transpose())
    return [res_labs, ans, res]


data = imageLoader.load_images_from_folder(args.folder_path)
if data is not None:
    model = modelFactory.get_model(args.feature_model)
    images = data[1]
    labels = data[0]
    features = model.compute_features_for_images(images)
    print(features.shape, "features")
    type_mat = create_type_type(features, labels)
    labels = type_mat[0]
    feature_type_mat = type_mat[2]
    print(feature_type_mat.shape, "feature_type")
    type_mat = type_mat[1]
    print(type_mat.shape)
    file_name = "latent_semantics_" + args.feature_model + "_" + args.tech + "_type_" + str(args.k) + ".json"
    if args.tech == 'pca':
        # PCA
        args.tech
    elif args.tech == 'svd':
        svd = SVD(args.k)
        latent_data = [labels, svd.compute_semantics(type_mat), type_mat.tolist(), feature_type_mat.tolist()]
        print_semantics_type(labels, np.matmul(np.array(latent_data[1][0]), np.array(latent_data[1][1])))
        save_features_to_json(args.folder_path, latent_data, file_name)
    elif args.tech.lower() == 'lda':
        lda = LDA(args.k)
        lda.compute_semantics(type_mat)
        latent_data = lda.transform_data(type_mat)
        print_semantics_type(labels, latent_data)
        lda.save_model(file_name)
        save_features_to_json(args.folder_path, [labels, latent_data.tolist()], file_name)
    else:
        kmeans = KMeans(args.k)
        kmeans.compute_semantics(type_mat)
        latent_data = kmeans.transform_data(type_mat)
        print_semantics_type(labels, latent_data)
        kmeans.save_model(file_name)
        save_features_to_json(args.folder_path, [labels, latent_data.tolist()], file_name)
