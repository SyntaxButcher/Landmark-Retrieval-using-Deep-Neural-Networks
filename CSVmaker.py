import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications import ResNet101V2
import pickle
from tqdm import tqdm
import csv
import random
import matplotlib.pyplot as plt

class ArcFace(layers.Layer):
    def __init__(self, num_classes=10, margin=0.5, scale=64, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.num_classes),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs, labels=None):
        cosine = tf.matmul(tf.nn.l2_normalize(inputs, axis=1), tf.nn.l2_normalize(self.W, axis=0))
        if labels is None:
            return self.scale * cosine
        else:
            one_hot = tf.one_hot(labels, depth=self.num_classes)
            theta = tf.acos(tf.clip_by_value(cosine, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
            marginal_cosine = tf.cos(theta + self.margin)
            hard_example = tf.cast(tf.greater(cosine, one_hot - self.margin), tf.float32)
            cos_m = tf.where(one_hot == 1, marginal_cosine, cosine - one_hot * self.margin)
            output = self.scale * tf.where(one_hot == 1, cos_m, hard_example * (cosine - self.margin * one_hot))
            return output
        
def ResNet101(input_shape, num_classes):
    pretrained_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in pretrained_model.layers:
        layer.trainable = False

    x = pretrained_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)

    predictions = ArcFace(num_classes=num_classes, margin=0.3, scale=46)(x)

    return Model(pretrained_model.input, predictions)

# Load the saved model weights
input_shape = (300, 300, 3)
num_classes = 81313
model = ResNet101(input_shape, num_classes)
model_weights_path = "model_weights_8_epoch.h5"
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print("Weights loaded!")

# Function to preprocess and predict the class of the given image
def predict_image_class(image_name, model):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(script_dir, 'oxford')
    img_path = os.path.join(img_folder, image_name)
    img = kimage.load_img(img_path, target_size=(300, 300))
    img_array = kimage.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class probabilities and return the class with highest probability
    class_probabilities = model.predict(img_array)
    predicted_class = np.argmax(class_probabilities, axis=1)
    
    return predicted_class[0]

def create_class_dictionary(image_folder, model):
    class_dict = {}
    for image_name in tqdm(os.listdir(image_folder), desc="Creating class dictionary"):
        img_class = predict_image_class(os.path.join(image_folder, image_name), model)
        if img_class in class_dict:
            class_dict[img_class].append(image_name)
        else:
            class_dict[img_class] = [image_name]
    return class_dict

# Find the class of the given image
def find_image_class(image_name, class_dict):
    for img_class, images in class_dict.items():
        if image_name in images:
            return img_class
    return None

def retrieve_similar_images(query_img_path, model, loaded_class_dict, n=10):
    # Find the class of the query image
    query_image_name = os.path.basename(query_img_path)
    query_image_class = find_image_class(query_image_name, loaded_class_dict)

    if query_image_class is not None:
        # Get the list of images in the same class and remove the query image from the list
        matching_images = [img for img in loaded_class_dict[query_image_class] if img != query_image_name]

        # Randomly select n images from the matching images
        random.shuffle(matching_images)
        top_n_images = matching_images[:n]

        return top_n_images
    else:
        return []

# oxford_image_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oxford')
# class_dict = create_class_dictionary(oxford_image_folder, model)

pickle_file_path = 'OxfordImagesClass.pickle'
# with open(pickle_file_path, 'wb') as handle:
#     pickle.dump(class_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pickle_file_path, 'rb') as handle:
    loaded_class_dict = pickle.load(handle)

# Load the data from the pickle file
with open('oxfordGT.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

# # Create the prediction CSV file
# output_csv_path = 'predictions.csv'
# with open(output_csv_path, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # Write the header row
#     csv_writer.writerow(['query_image', 'matching_images'])
#     # Iterate through the qimlist
#     for query_image in ground_truth['qimlist']:

#         # Find the class of the query image
#         query_image = query_image + '.jpg'
#         query_image_class = find_image_class(query_image, loaded_class_dict)

#         if query_image_class is not None:
#             # Get the list of matching images and remove the query image from the list
#             matching_images = [img for img in loaded_class_dict[query_image_class] if img != query_image]

#             # Write the row to the CSV file
#             csv_writer.writerow([query_image, ' '.join(matching_images)])
#         else:
#             # Write the row with an empty second column
#             csv_writer.writerow([query_image, ''])

# Create the ground truth CSV file
def create_ground_truth_csv(ground_truth, csv_file_name):
    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['query', 'matches'])

        for query_index, query in enumerate(ground_truth['qimlist']):
            gnd = ground_truth['gnd'][query_index]
            # hard_matches = gnd.get('hard', [])
            easy_matches = gnd.get('easy', [])
            all_matches = easy_matches

            # Get the matching image names using their indexes
            matching_images = ' '.join([ground_truth['imlist'][idx] for idx in all_matches])

            writer.writerow([query, matching_images])

# csv_file_name = 'groundTruth.csv'
# create_ground_truth_csv(ground_truth, csv_file_name)

def retrieve_similar_images(query_img_path, model, loaded_class_dict, n=10):
    # Find the class of the query image
    query_image_name = os.path.basename(query_img_path)
    query_image_class = find_image_class(query_image_name, loaded_class_dict)

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
    fig, ax = plt.subplots(2, n_images + 1, figsize=(15, 4))

    # Display the original query image in the first row
    query_img = kimage.load_img(query_img_path, target_size=(300, 300))
    ax[0, 0].imshow(query_img)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Query Image')

    # Hide the empty axes in the first row
    for i in range(1, n_images + 1):
        ax[0, i].axis('off')

    # Display the matching images in the second row
    for i, img_name in enumerate(similar_images):
        img_path = os.path.join('oxford', img_name)
        img = kimage.load_img(img_path, target_size=(300, 300))
        ax[1, i].imshow(img)
        ax[1, i].axis('off')
        ax[1, i].set_title(f'Matching Image {i + 1}')

    plt.tight_layout()
    plt.show()

# Example usage
query_img_path = 'oxford/bodleian_000107.jpg'
top_n = 5

similar_images = retrieve_similar_images(query_img_path, model, loaded_class_dict, top_n)
print("Top similar images:")
for img_name in similar_images:
    print(img_name)

similar_images = retrieve_similar_images(query_img_path, model, loaded_class_dict, top_n)
display_images(query_img_path, similar_images)