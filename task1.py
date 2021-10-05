import imageLoader
import modelFactory
import argparse

parser = argparse.ArgumentParser(description="Task 1")
parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
)

args = parser.parse_args()

image = imageLoader.load_image(args.image_path)
if image is not None:
    model = modelFactory.get_model(args.model)
    model.visualize_feature(image)
