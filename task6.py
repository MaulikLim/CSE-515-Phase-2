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
from tech.LDA import LDA
import tech.LDAHelper as lda_helper
from utilities import intersection_similarity_between_features, print_semantics_type

parser = argparse.ArgumentParser(description="Task 6")
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
    "-q",
    "--image_path",
    type=str,
    required=True,
)
args = parser.parse_args()


def getType(result, isDist):
    scores = {}
    for res in result:
        label = res[0].split("-")[1]
        if label not in scores:
            scores[label] = 0
        scores[label] += res[1]
    ans = ""
    maxScore = 0
    for x, y in scores.items():
        if ans == "" or (not isDist and y > maxScore) or (isDist and y < maxScore):
            maxScore = y
            ans = x
    return ans


data = imageLoader.load_images_from_folder(args.folder_path)
if data is not None:
    l_features = load_json(args.latent_path)
    file_name = args.latent_path.split("/")[-1]
    info = file_name.split("_")
    model = modelFactory.get_model(info[2])
    tech = info[3]
    type = info[4]
    l_k = info[5]
    if tech == 'pca':
        # PCA
        pass
    elif tech == 'svd':

        labels = data[0]
        r_mat = np.array(l_features[1][2]).transpose()
        if type == 'type' or type == 'subject':
            feature_type_mat = np.array(l_features[3])
        new_data = []
        for d in data[1]:
            feature_mat = model.compute_features(d)
            if type == 'type' or type == 'subject':
                feature_mat = np.matmul(feature_mat, feature_type_mat.T)
            l_feature_mat = np.matmul(feature_mat, r_mat)
            new_data.append(l_feature_mat)
        q_feature_mat = model.compute_features(imageLoader.load_image(args.image_path))
        if type == 'type' or type == 'subject':
            q_feature_mat = np.matmul(q_feature_mat, feature_type_mat.T)
        l_q_feature_mat = np.matmul(q_feature_mat, r_mat)
        result = []
        for ind,d in enumerate(new_data):
            sim_score = np.sum(np.abs(d-l_q_feature_mat))
            result.append([labels[ind],sim_score])
        print(getType(result,True))
    elif tech=='lda':
        l_features = load_json(args.latent_path)
        print(len(l_features))
        lda = LDA(file_name=file_name)
        labels = data[0]
        original_metrics = model.compute_features_for_images(data[1])
        if type == 'type' or type == 'subject':
            transform_matrix = np.array(l_features[2]).T
            original_metrics = np.matmul(original_metrics, transform_matrix)
            print(original_metrics.shape, end='')
            print('intermediate transformation')
        else:
            original_metrics = lda_helper.transform_cm_for_lda(original_metrics)
        original_metrics = lda.transform_data(original_metrics)
        print(original_metrics.shape, end='')
        print('final transformation')

        q_feature_mat = model.compute_features(imageLoader.load_image(args.image_path))
        q_feature_mat = q_feature_mat.reshape([1, q_feature_mat.shape[0]])
        if type == 'type' or type == 'subject':
            q_feature_mat = np.matmul(q_feature_mat, transform_matrix)
        else:
            q_feature_mat = lda_helper.transform_cm_for_lda(q_feature_mat)
        print(q_feature_mat.shape, end='')
        print('query intermediate transformation')

        q_feature_mat = lda.transform_data(q_feature_mat)
        print(q_feature_mat.shape, end='')
        print('query final transformation')
        result = []
        for ind, d in enumerate(original_metrics):
            sim_score = np.sum(np.abs(d - q_feature_mat))
            result.append([labels[ind], sim_score])
        print(getType(result,True))
        result = sorted(result, key=lambda x: x[1])[:12]
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "Distance score:", ele[1])
    else:
        pass
