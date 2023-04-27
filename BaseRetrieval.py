import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
import random
from math import ceil

def load_image(img_path, target_size=(300, 300)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def find_image_class(image_name, class_dict):
    for class_name, images in class_dict.items():
        if image_name in images:
            return class_name
    return None

def retrieve_similar_images(query_img_path, model, loaded_class_dict, n=10):
    # Find the class of the query image
    query_image_name = os.path.basename(query_img_path)
    query_image_class = find_image_class(query_image_name, loaded_class_dict)
    print(query_image_class)

    if query_image_class is not None:
        # Get the list of images in the same class and remove the query image from the list
        matching_images = [img for img in loaded_class_dict[query_image_class] if img != query_image_name]

        # Randomly select n images from the matching images
        random.shuffle(matching_images)
        top_n_images = matching_images[:n]

        return top_n_images
    else:
        return []

def display_images(query_img_path, similar_images):
    n_images = len(similar_images)
    n_rows = ceil(n_images / 5) + 1  # Add 1 for the query image row

    fig, ax = plt.subplots(n_rows, 6, figsize=(15, 4 * n_rows))

    # Display the original query image in the first row
    query_img = image.load_img(query_img_path, target_size=(300, 300))
    ax[0, 0].imshow(query_img)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Query Image')

    # Hide the empty axes in the first row
    for i in range(1, 6):
        ax[0, i].axis('off')

    # Display the matching images in subsequent rows, 5 images per row
    for i, img_name in enumerate(similar_images):
        img_path = os.path.join('oxford', img_name)
        img = image.load_img(img_path, target_size=(300, 300))
        row, col = divmod(i, 5)
        ax[row + 1, col].imshow(img)
        ax[row + 1, col].axis('off')
        ax[row + 1, col].set_title(f'Matching Image {i + 1}')

    plt.tight_layout()
    plt.show()

query_img_path = 'oxford/all_souls_000013.jpg'
model_weights_path = 'model_weights_8_epoch.h5'
oxford_images_class_path = 'OxfordImagesClass.pickle'
top_k = 10

with open(oxford_images_class_path, 'rb') as f:
    loaded_class_dict = pickle.load(f)

similar_images = retrieve_similar_images(query_img_path, model_weights_path, loaded_class_dict, top_k)
display_images(query_img_path, similar_images)
print("Top similar images:")
for img_name in similar_images:
    print(img_name)
