import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # raise NotImplementedError
    images = []
    labels = []

    # Iterate over category directories (0 to NUM_CATEGORIES - 1)
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))  # Get category folder path

        # Ensure the directory exists
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} directory not found. Skipping...")
            continue

        # Iterate over image files in the category directory
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)

            # Read image using OpenCV
            img = cv2.imread(file_path)

            if img is None:
                print(f"Warning: Unable to read {file_path}. Skipping...")
                continue

            # Resize image to the required dimensions
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Append image and label to lists
            images.append(img_resized)
            labels.append(category)

    return images, labels



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # raise NotImplementedError
    model = Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Layer
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flattening layer
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # Helps prevent overfitting

        # Output Layer (NUM_CATEGORIES classes, using softmax for classification)
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
