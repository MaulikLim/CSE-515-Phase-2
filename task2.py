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
    "-y",
    "--Y",
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

def extract_type_weight_pairs(labels, metrics):
  type_metrics = {}
  for x in range(len(labels)):
    image_type = labels[x].split("-")[1]
    type_data = []
    if image_type in type_metrics:
      type_data = type_metrics[image_type]
    type_data.append(metrics[x])
    type_metrics[image_type] = type_data
  y = metrics.shape[1]
  type_weights = []
  types = []
  index = 0
  for image_type, data in type_metrics.items():
    types.append(image_type)
    count = 0
    type_weight = np.zeros(y)
    for d in data:
      type_weight += d
      count += 1
    type_weights.append(type_weight/count)
    index += 1
  return [types, type_weights]

def print_semantics(labels, metrics):
    subjects, subject_weights = extract_type_weight_pairs(labels, metrics)
    subject_weights = np.array(subject_weights)
    for x in range(subject_weights.shape[1]):
        print("latent semantic "+str(x)+":", end=" ")
        semantic_weights = subject_weights[:,x:x+1].flatten()
        sorted_order = np.flip(np.argsort(semantic_weights))
        for x in sorted_order:
            print(subjects[x]+"="+str(semantic_weights[x]), end=" ")
        print()

data = imageLoader.load_images_from_folder(args.folder_path,'*',args.Y)
if data is not None:
    model = modelFactory.get_model(args.feature_model)
    images = data[1]
    labels = data[0]
    if(args.tech=='pca'):
        #PCA
        args.tech
    elif(args.tech=='svd'):
        svd = SVD(model,args.k)
        latent_data = svd.compute_semantics_type(images)
        print("done.")
    elif(args.tech=='lda'):
        args.tech
    else:
        args.tech

