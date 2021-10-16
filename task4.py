from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import os
import numpy as np
import datetime
from tech.SVD import SVD
from tech.LDA import LDA
from utilities import print_semantics_sub

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

def create_sub_sub(features, labels):
    res = []
    labs = []
    res_labs = []
    for i in range(len(labels)):
        lab = labels[i].split("-")[2]
        if lab not in labs:
            labs.append(lab)
            res_labs.append(labels[i])
            res.append(features[i])
        else:
            ind = labs.index(lab)
            res[ind] = np.mean( np.array([ res[ind], features[i] ]), axis=0 )
    res = np.array(res)
    ans = np.matmul(res,res.transpose())
    return [res_labs,ans,res]

data = imageLoader.load_images_from_folder(args.folder_path)
if data is not None:
    model = modelFactory.get_model(args.feature_model)        
    images = data[1]
    labels = data[0]
    features = model.compute_features_for_images(images)
    sub_mat = create_sub_sub(features,labels)
    labels = sub_mat[0]
    feature_sub_mat = sub_mat[2]
    sub_mat = sub_mat[1]
    if(args.tech=='pca'):
        #PCA
        args.tech
    elif(args.tech=='svd'):
        svd = SVD(args.k)
        latent_data = [labels, svd.compute_semantics(sub_mat), sub_mat.tolist(), feature_sub_mat.tolist()]
        print_semantics_sub(labels,np.matmul(np.array(latent_data[1][0]),np.array(latent_data[1][1])))
        file_name = "latent_semantics_"+args.feature_model+"_"+args.tech+"_subject_"+str(args.k)+".json"
        save_features_to_json(args.folder_path,latent_data,file_name)
    elif(args.tech=='lda'):
        lda = LDA(k=args.k)
        lda.compute_semantics(sub_mat)
        latent_data = lda.transform_data(sub_mat)
        print_semantics_sub(labels, latent_data)
        file_name = "latent_semantics_"+args.feature_model+"_"+args.tech+"_subject_"+str(args.k)
        lda.save_model(file_name)
        save_features_to_json(args.folder_path, [labels, latent_data.tolist()], file_name)
    else:
        args.tech

