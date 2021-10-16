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
        args.tech
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
        for ind, d in enumerate(new_data):
            sim_score = np.sum(np.abs(d - l_q_feature_mat))
            result.append([labels[ind], sim_score])
        result = sorted(result, key=lambda x: x[1])[:args.k]
        i = 0
        for ele in result:
            i += 1
            print(i, ele[0], "Similarity score::", ele[1])
            imageLoader.show_image(os.path.join(args.folder_path, ele[0]))
    elif tech == 'lda':
        pass
    else:
        pass
