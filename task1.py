from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb

from tech.PCA import PCA
from tech.SVD import SVD
from tech.LDA import LDA
from tech.KMeans import KMeans
import tech.LDAHelper as lda_helper
from utilities import print_semantics_sub, print_semantics_type

parser = argparse.ArgumentParser(description="Task 1")
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
    "-x",
    "--X",
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

data = imageLoader.load_spec_images_from_folder(args.folder_path, args.X, '*')
if data is not None:
    model = modelFactory.get_model(args.feature_model)
    images = data[1]
    labels = data[0]
    data = model.compute_features_for_images(images)
    file_name = "latent_semantics_" + args.feature_model + "_" + args.tech + "_" + args.X + "_" + str(args.k) + ".json"
    if args.tech.lower() == 'pca':
        pca = PCA(args.k)
        latent_data = [labels, pca.compute_semantics(data)]
        print_semantics_sub(labels,np.matmul(data,np.array(latent_data[1][0])))
        save_features_to_json(args.folder_path,latent_data,file_name)
    elif args.tech.lower() == 'svd':
        svd = SVD(args.k)
        latent_data = [labels, svd.compute_semantics(data)]
        print_semantics_sub(labels, np.matmul(np.array(latent_data[1][0]), np.array(latent_data[1][1])))
        save_features_to_json(args.folder_path, latent_data, file_name)
    elif args.tech.lower() == 'lda':
        lda = LDA(args.k)
        data = lda_helper.transform_cm_for_lda(data)
        lda.compute_semantics(data)
        latent_data = lda.transform_data(data)
        print_semantics_sub(labels, latent_data)
        lda.save_model(file_name)
        save_features_to_json(args.folder_path, [labels, latent_data.tolist()], file_name)
    else:
        kmeans = KMeans(args.k)
        kmeans.compute_semantics(data)
        latent_data = kmeans.transform_data(data)
        print_semantics_sub(labels, np.array(latent_data), do_rev=True)
        save_features_to_json(
            args.folder_path,
            [labels, latent_data, kmeans.centroids.tolist()],
            file_name
        )
