import numpy as np
from genericpath import isfile
import os
import json


# Loads json file of feature descriptor
def load_json(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as feature_descriptors:
            return json.load(feature_descriptors)
    return None

# Retrieves numpy array of labels and their corresponding feature vectors for the given model name
# def load_features_for_model(folder_path, model_name):
#     print('Loading features from path '+folder_path)
#     features_descriptors = load_json(folder_path)
#     if features_descriptors is not None:
#         return np.array(features_descriptors['labels']), np.array(features_descriptors[model_name])
#     return [None, None]
