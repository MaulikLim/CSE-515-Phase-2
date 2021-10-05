import os
import cv2 as cv
import matplotlib.pyplot as plt

# Loads image present at the given path, which is relative to the current location
def load_image(path):
    image_path = os.path.join(os.getcwd(), path)
    image = cv.imread(image_path, cv.COLOR_BGR2GRAY)
    if image is not None:
        return image
    print("No image found for given path "+image_path)

def show_image(path):
    img = load_image(path)
    plt.imshow(img,cmap='gray', vmin=0, vmax=255)
    plt.show()
    
# Loads images from the fodler at the given path, relative to the current location
def load_images_from_folder(path):
    images = []
    labels = []
    folder_path = os.path.join(os.getcwd(), path)
    if not os.path.isdir(path):
        print("No directory found for "+folder_path)
        return
    for filename in os.listdir(folder_path):
        img = cv.imread(os.path.join(folder_path, filename), cv.COLOR_BGR2GRAY)
        if img is not None:
            # print(filename)
            labels.append(filename)
            images.append(img)
    return [labels, images]
