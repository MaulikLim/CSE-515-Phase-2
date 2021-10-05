import imageLoader, modelFactory, featureLoader
import argparse
import featureGenerator
import os

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
    "-m",
    "--model",
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
# loads labels and feature vectors for the given folder path and model name
labels, features = featureLoader.load_features_for_model(folder_path, args.model)

# if no features found, it generates new features
if features is None or len(features) == 0:
    print("No feature descriptors found. Generating new ones")
    labels, images = imageLoader.load_images_from_folder(args.folder_path)
    featureGenerator.generate_and_save_features(labels, images, args.folder_path)
    labels, features = featureLoader.load_features_for_model(folder_path, args.model)

# retrieves top k similar images for the given model, prints their similarity score and visualizes them
model = modelFactory.get_model(args.model)
ans = model.get_top_k(labels, features, args.image_id, args.k)
for x in ans:
    print(x)
    image_path = os.path.join(folder_path, x[0])
    imageLoader.show_image(image_path)

# acc = 0
# for name in labels:
#     result = model.get_top_k(labels, features, name, args.k)
#     match = 0
#     for label, score in result:
# #             print(img);
#         if(label.split("_")[0]==name.split("_")[0]):
#             match += 1
#     acc += (match/args.k)
# print(acc)