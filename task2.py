import imageLoader
import argparse
import featureGenerator

parser = argparse.ArgumentParser(description="Task 2")
parser.add_argument(
    "-f",
    "--folder_path",
    type=str,
    required=True,
)

args = parser.parse_args()
labels, images = imageLoader.load_images_from_folder(args.folder_path)


if len(images) > 0:
    featureGenerator.generate_and_save_features(labels, images, args.folder_path)
