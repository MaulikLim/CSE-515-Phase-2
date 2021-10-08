from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import os
import numpy as np
import datetime
from tech.SVD import SVD
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
    for i in range(len(labels)):
        lab = labels[i].split("-")[1]
        if lab not in labs:
            labs.append(lab)
            res.append(features[i])
        else:
            ind = labs.index(lab)
            res[ind] = np.mean( np.array([ res[ind], features[i] ]), axis=0 )
    return [labs,res]
data = imageLoader.load_images_from_folder(args.folder_path)
if data is not None:
    model = modelFactory.get_model(args.feature_model)        
    images = data[1]
    labels = data[0]
    features = model.compute_features_for_images(images)
    type_mat = create_type_type(features,labels)
    labels = type_mat[0]
    type_mat = type_mat[1]
    if(args.tech=='pca'):
        #PCA
        args.tech
    elif(args.tech=='svd'):
        svd = SVD(args.k)
        latent_data = [labels, svd.compute_semantics(type_mat)]
        print_semantics_type(labels,np.matmul(np.array(latent_data[0]),np.array(latent_data[1])))
        file_name = "latent_semantics_"+args.feature_model+"_"+args.tech+"_type_"+str(args.k)+".json"
        save_features_to_json(args.folder_path,latent_data,file_name)
        # print_semantics(labels,np.matmul(np.array(latent_data[1][0]),np.array(latent_data[1][1])))
        # json_feature_descriptors = json.dumps(latent_data, indent=4)
        # file_name = "latent_semantics_"+args.feature_model+"_"+args.tech+"_"+args.X+"_"+str(args.k)+".json"
        # file_path = os.path.join(args.folder_path, file_name)
        # if os.path.isfile(file_path):
        #         os.remove(file_path)
        # with open(file_path, "w") as out_file:
        #     out_file.write(json_feature_descriptors)
        print("done.")
    elif(args.tech=='lda'):
        args.tech
    else:
        args.tech

