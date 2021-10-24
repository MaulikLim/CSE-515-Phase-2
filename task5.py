import pickle

from featureGenerator import save_features_to_json
from featureLoader import load_json
import imageLoader
import modelFactory
import argparse
import json
import os
import numpy as np
import datetime
import featureLoader
from tech.SVD import SVD
from tech.LDA import LDA
import tech.LDAHelper as lda_helper
from utilities import intersection_similarity_between_features, print_semantics_type

parser = argparse.ArgumentParser(description="Task 5")
parser.add_argument(
    "-fp",
    "--folder_path",
    type=str,
    required=True,
)
parser.add_argument(
    "-lp",
    "--latent_path",
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
    "-q",
    "--image_path",
    type=str,
    required=True,
)
args = parser.parse_args()

file_name = args.latent_path.split("/")[-1]
info = file_name.split("_")
model = modelFactory.get_model(info[2])
tech = info[3].lower()
type = info[4]
l_k = info[5]
data = featureLoader.load_features_for_model(args.folder_path, info[2])
if data is not None:
    if tech == 'pca':
        l_features = load_json(args.latent_path)
        labels = data[0]
        l_mat = np.array(l_features[1][0])
        # if type == 'type' or type == 'subject':
        #     feature_type_mat = np.array(l_features[3])
        # new_data = []
        # for d in data[1]:
        #     feature_mat = model.compute_features(d)
        #     if type == 'type' or type == 'subject':
        #         feature_mat = np.matmul(feature_mat, feature_type_mat.T)
        #     l_feature_mat = np.matmul(feature_mat, l_mat)
        #     new_data.append(l_feature_mat)
        feature_mat = data[1]
        if type == 'type' or type == 'subject':
            feature_type_mat = np.array(l_features[3])
            feature_mat = np.matmul(feature_mat, feature_type_mat.T)
        new_data = np.matmul(feature_mat, l_mat)
        q_feature_mat = model.compute_features(imageLoader.load_image(args.image_path))
        if type == 'type' or type == 'subject':
            q_feature_mat = np.matmul(q_feature_mat, feature_type_mat.T)
        l_q_feature_mat = np.matmul(q_feature_mat, l_mat)
        result = []
        if info[2] == 'cm':
            for ind, d in enumerate(new_data):
                sim_score = np.sum(np.abs(d - l_q_feature_mat))
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[:args.k]
        elif info[2] == 'elbp':
            for ind, d in enumerate(new_data):
                sim_score = intersection_similarity_between_features(d , l_q_feature_mat)
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[::-1][:args.k]
        else:
            for ind, d in enumerate(new_data):
                sim_score = intersection_similarity_between_features(d , l_q_feature_mat)
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[::-1][:args.k]    
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "score:", ele[1])
            imageLoader.show_image(os.path.join(args.folder_path, ele[0]))
    elif tech == 'svd':
        l_features = load_json(args.latent_path)
        labels = data[0]
        r_mat = np.array(l_features[1][2]).transpose()
        # if type == 'type' or type == 'subject':
        #     feature_type_mat = np.array(l_features[3])
        # new_data = []
        # for d in data[1]:
        #     feature_mat = model.compute_features(d)
        #     if type == 'type' or type == 'subject':
        #         feature_mat = np.matmul(feature_mat, feature_type_mat.T)
        #     l_feature_mat = np.matmul(feature_mat, r_mat)
        #     new_data.append(l_feature_mat)
        feature_mat = data[1]
        if type == 'type' or type == 'subject':
            feature_type_mat = np.array(l_features[3])
            feature_mat = np.matmul(feature_mat, feature_type_mat.T)
        new_data = np.matmul(feature_mat, r_mat)
        q_feature_mat = model.compute_features(imageLoader.load_image(args.image_path))
        if type == 'type' or type == 'subject':
            q_feature_mat = np.matmul(q_feature_mat, feature_type_mat.T)
        l_q_feature_mat = np.matmul(q_feature_mat, r_mat)
        result = []
        if info[2] == 'cm':
            for ind, d in enumerate(new_data):
                sim_score = np.sum(np.abs(d - l_q_feature_mat))
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[:args.k]
        elif info[2] == 'elbp':
            for ind, d in enumerate(new_data):
                sim_score = intersection_similarity_between_features(d , l_q_feature_mat)
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[::-1][:args.k]
        else:
            for ind, d in enumerate(new_data):
                sim_score = intersection_similarity_between_features(d , l_q_feature_mat)
                result.append([labels[ind], sim_score])
            result = sorted(result, key=lambda x: x[1])[::-1][:args.k]
            
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "score:", ele[1])
            imageLoader.show_image(os.path.join(args.folder_path, ele[0]))
    elif tech == 'lda':
        l_features = load_json(args.latent_path)
        lda = LDA(file_name=args.folder_path + '/' + file_name)
        labels = data[0]
        # original_metrics = model.compute_features_for_images(data[1])
        original_metrics = data[1]
        if type == 'type' or type == 'subject':
            transform_matrix = np.array(l_features[2]).T
            original_metrics = np.matmul(original_metrics, transform_matrix)
        else:
            original_metrics = lda_helper.transform_cm_for_lda(original_metrics)
        original_metrics = lda.transform_data(original_metrics)

        q_feature_mat = model.compute_features(imageLoader.load_image(args.image_path))
        q_feature_mat = q_feature_mat.reshape([1, q_feature_mat.shape[0]])
        if type == 'type' or type == 'subject':
            q_feature_mat = np.matmul(q_feature_mat, transform_matrix)
        else:
            q_feature_mat = lda_helper.transform_cm_for_lda(q_feature_mat)

        q_feature_mat = lda.transform_data(q_feature_mat)
        result = []
        for ind, d in enumerate(original_metrics):
            sim_score = np.linalg.norm(d - q_feature_mat)
            result.append([labels[ind], sim_score])
        result = sorted(result, key=lambda x: x[1])[:args.k]
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "Distance score:", ele[1])
            imageLoader.show_image(os.path.join(args.folder_path, ele[0]))
    else:
        l_features = load_json(args.latent_path)
        # with open(args.latent_path, 'rb') as f:
        centroids = l_features[2]
        labels = data[0]
        feature_type_mat = []
        if type == 'type' or type == 'subject':
            feature_type_mat = np.array(l_features[3])
        data_cluster_index, data_in_latent_space, data_cluster_dist = [], [], []
        for feature_mat in data[1]:
            # feature_mat = model.compute_features(d)
            if type == 'type' or type == 'subject':
                feature_mat = np.matmul(feature_mat, feature_type_mat.T)
            min_dist_arr = np.sum((centroids - feature_mat) ** 2, axis=1)
            # min_dist_ctr = np.argmin(min_dist_arr)
            # data_cluster_index.append(min_dist_ctr)
            # data_cluster_dist.append(np.min(min_dist_arr))
            data_in_latent_space.append(min_dist_arr)
        # data_cluster_index = np.array(data_cluster_index)
        # data_in_latent_space = np.zeros((data_cluster_index.size, centroids.shape[0]))
        # data_in_latent_space[np.arange(data_cluster_index.size), data_cluster_index] = data_cluster_dist

        query_features = model.compute_features(imageLoader.load_image(args.image_path))
        if type == 'type' or type == 'subject':
            query_features = np.matmul(query_features, feature_type_mat.T)
        min_dist_arr = np.sum((centroids - query_features) ** 2, axis=1)
        # min_dist_ctr = np.argmin(min_dist_arr)
        # query_in_latent_space = np.zeros(centroids.shape[0])
        # query_in_latent_space[min_dist_ctr] = np.min(min_dist_arr)
        query_in_latent_space = min_dist_arr

        result = []
        for ind, d in enumerate(data_in_latent_space):
            sim_score = np.sum(np.linalg.norm(d - query_in_latent_space))
            result.append([labels[ind], sim_score])

        result = sorted(result, key=lambda x: x[1])[:args.k]
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "Distance score:", ele[1])
            imageLoader.show_image(os.path.join(args.folder_path, ele[0]))
