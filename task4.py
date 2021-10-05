import imageLoader, modelFactory, featureLoader
import argparse
import featureGenerator
import os
import numpy as np

parser = argparse.ArgumentParser(description="Task 3")
parser.add_argument(
    "-f",
    "--folder_path",
    type=str,
    required=True,
)
parser.add_argument(
    "-i",
    "--image_id",
    type=str,
    required=True,
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    required=True,
)


args = parser.parse_args()
folder_path = os.path.join(os.getcwd(), args.folder_path)
models = modelFactory.get_all_models()
# Loads labels and features to check if features are already computed
labels, features = featureLoader.load_features_for_model(folder_path, models[0].name)

# If no features are found, it generates new descriptors
if features is None or len(features) == 0:
    print("No feature descriptors found. Generating new ones")
    labels, images = imageLoader.load_images_from_folder(args.folder_path)
    featureGenerator.generate_and_save_features(labels, images, args.folder_path)
    labels, features = featureLoader.load_features_for_model(folder_path, models[0].name)

# Fetches normalized similarity scores for each model and for each image with the query image
models = modelFactory.get_all_models()
similarity_scores = []
normalizedSimilarities = np.zeros(features.shape[0])
for model in models:
    labels, features = featureLoader.load_features_for_model(folder_path, model.name)
    model_similarity_scores = model.get_normalized_similarities(labels, features, args.image_id)
    similarity_scores.append(model_similarity_scores)
    normalizedSimilarities += model_similarity_scores

# Displays the details about top k similar images
ans = np.flipud(np.argsort(normalizedSimilarities)[-args.k-1:])
for x in ans:
    print(labels[x], end=" ")
    print("Overall similarity: " + str(normalizedSimilarities[x]), end=" ")
    for y in range(len(models)):
        print(models[y].name + ": " + str(similarity_scores[y][x]), end=" ")
    image_path = os.path.join(folder_path, labels[x])
    imageLoader.show_image(image_path)
    print()