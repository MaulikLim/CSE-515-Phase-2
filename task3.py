from featureGenerator import save_features_to_json
import imageLoader
import featureLoader
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


# def create_type_type():
#     res = []
#     labs = []
#     res_labs = []
#     for i in range(len(labels)):
#         lab = labels[i].split("-")[1]
#         if lab not in labs:
#             labs.append(lab)
#             res_labs.append(labels[i])
#             res.append(features[i])
#         else:
#             ind = labs.index(lab)
#             res[ind] = np.mean(np.array([res[ind], features[i]]), axis=0)
#     res = np.array(res)
#     print(res.shape, "res")
#     ans = np.matmul(res, res.transpose())
#     return [res_labs, ans, res]

def create_type_type(metrics, labels):
    type_metrics = {}
    for x in range(len(labels)):
        image_type = labels[x].split("-")[1]
        type_data = []
        if image_type in type_metrics:
            type_data = type_metrics[image_type]
        type_data.append(metrics[x])
        type_metrics[image_type] = type_data
    y = metrics.shape[1]
    type_features = []
    types = []
    index = 0
    for image_type, data in type_metrics.items():
        types.append(image_type)
        count = 0
        type_weight = np.zeros(y)
        for d in data:
            type_weight += d
            count += 1
        type_features.append(type_weight/count)
        index += 1
    type_features = np.array(type_features)
    type_type = np.matmul(type_features, type_features.T)
    return [types, type_type, type_features]


data = featureLoader.load_features_for_model(args.folder_path, args.feature_model)
if data is not None:
    # model = modelFactory.get_model(args.feature_model)
    features = data[1]
    labels = data[0]
    # features = model.compute_features_for_images(images)
    type_mat = create_type_type(features, labels)
    labels = type_mat[0]
    feature_type_mat = type_mat[2]
    type_mat = type_mat[1]
    type_type_file_name = "type_type_"+args.feature_model+".json"
    
    save_features_to_json(args.folder_path, [labels,type_mat.tolist()], type_type_file_name)

    file_name = "latent_semantics_" + args.feature_model + \
        "_" + args.tech + "_type_" + str(args.k) + ".json"
    if args.tech == 'pca':
        # PCA
        args.tech
    elif args.tech == 'svd':
        svd = SVD(args.k)
        latent_data = [labels, svd.compute_semantics(
            type_mat), type_mat.tolist(), feature_type_mat.tolist()]
        print_semantics_type(labels, np.matmul(
            np.array(latent_data[1][0]), np.array(latent_data[1][1])))
        save_features_to_json(args.folder_path, latent_data, file_name)
    elif args.tech.lower() == 'lda':
        lda = LDA(args.k)
        lda.compute_semantics(type_mat)
        latent_data = lda.transform_data(type_mat)
        print_semantics_type(labels, latent_data)
        lda.save_model(file_name)
        save_features_to_json(args.folder_path, [
                              labels, latent_data.tolist(), feature_type_mat.tolist()], file_name)
    else:
        kmeans = KMeans(args.k)
        kmeans.compute_semantics(type_mat)
        latent_data = kmeans.transform_data(type_mat)
        print_semantics_type(labels, latent_data)
        # kmeans.save_model(file_name)
        save_features_to_json(
            args.folder_path,
            [labels, latent_data.tolist(), kmeans.centroids.tolist(), feature_type_mat.tolist()],
            file_name
        )
