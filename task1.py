import imageLoader
import modelFactory
import argparse
import json
import os

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

def extract_subject_weight_pairs(labels, metrics):
  subject_metrics = {}
  for x in range(len(labels)):
    subject = labels[x].split("-")[2]
    subject_data = []
    if subject in subject_metrics:
      subject_data = subject_metrics[subject]
    subject_data.append(metrics[x])
    subject_metrics[subject] = subject_data
  y = metrics.shape[1]
  subject_weights = []
  subjects = []
  index = 0
  for sub, data in subject_metrics.items():
    subjects.append(sub)
    count = 0
    subject_weight = np.zeros(y)
    for d in data:
      subject_weight += d
      count += 1
    subject_weights.append(subject_weight/count)
    index += 1
  return [subjects, subject_weights]

def print_semantics(labels, metrics):
    subjects, subject_weights = extract_subject_weight_pairs(labels, metrics)
    subject_weights = np.array(subject_weights)
    for x in range(subject_weights.shape[1]):
        print("latent semantic "+str(x)+":", end=" ")
        semantic_weights = subject_weights[:,x:x+1].flatten()
        sorted_order = np.flip(np.argsort(semantic_weights))
        for x in sorted_order:
            print(subjects[x]+"="+str(semantic_weights[x]), end=" ")
        print()

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
        latent_data = [labels, svd.compute_semantics_type(images)]
        json_feature_descriptors = json.dumps(latent_data, indent=4)
        file_name = "latent_semantics_"+args.feature_model+"_"+args.tech+"_"+args.X+"_"+str(args.k)+".json"
        file_path = os.path.join(args.folder_path, file_name)
        if os.path.isfile(file_path):
                os.remove(file_path)
        with open(file_path, "w") as out_file:
            out_file.write(json_feature_descriptors)
        print("done.")
    elif(args.tech=='lda'):
        args.tech
    else:
        args.tech

