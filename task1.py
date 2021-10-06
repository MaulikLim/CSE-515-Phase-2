import imageLoader
import modelFactory
import argparse

from tech.SVD import SVD

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

data = imageLoader.load_images_from_folder(args.folder_path,args.X,'*')
if data is not None:
    model = modelFactory.get_model(args.feature_model)
    images = data[1]
    labels = data[0]
    if(args.tech=='pca'):
        #PCA
        args.tech
    elif(args.tech=='svd'):
        svd = SVD(model,args.k)
        latent_data = svd.compute_semantics_type(images,labels)
        print("done.")
    elif(args.tech=='lda'):
        args.tech
    else:
        args.tech

