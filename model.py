import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Define the input image size
IMG_SIZE = (462, 300)

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    # Load the image
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{file_path}'")
        return None

    # Resize the image
    image = cv2.resize(image, IMG_SIZE)

    # Convert the image to a float32 data type and normalize the pixel values
    image = image.astype(np.float32) / 255.0

    return image

# Load an image
image_path = r'C:\Users\Abd El-Rahman\GitHub\Ai_Project\test\label\christchurch_285_vis.tif'
image = load_and_preprocess_image(image_path)

if image is not None:
    # Check the shape of the image
    print(image.shape)

    # Print the shape of the preprocessed image
    print(image.shape)
else:
    print("Image could not be loaded.")

# Define the paths to the training and testing data
TRAIN_IMAGE_DIR = 'train/image'
TRAIN_LABEL_DIR = 'train/label'
TEST_IMAGE_DIR = 'test/image'
TEST_LABEL_DIR = 'test/label'

# Load the training and testing data using the tf.data.Dataset API
train_image_ds = tf.data.Dataset.list_files(os.path.join(TRAIN_IMAGE_DIR, '*.tif'))
train_label_ds = tf.data.Dataset.list_files(os.path.join(TRAIN_LABEL_DIR, '*.tif'))
test_image_ds = tf.data.Dataset.list_files(os.path.join(TEST_IMAGE_DIR, '*.tif'))
test_label_ds = tf.data.Dataset.list_files(os.path.join(TEST_LABEL_DIR, '*.tif'))

# Load and preprocess the training and testing data
def load_and_preprocess_image_tf(file_path):
    # Load the image
    image = cv2.imread(file_path.numpy().decode(), cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{file_path.numpy().decode()}'")
        return None

    # Resize the image
    image = cv2.resize(image, IMG_SIZE)

    # Convert the image to a float32 data type and normalize the pixel values
    image = image.astype(np.float32) / 255.0

    return image

def load_and_preprocess_label(file_path):
    image = cv2.imread(file_path.numpy().decode(), cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32) / 255.0
    return image

def tf_wrapper_load_and_preprocess_image(file_path):
    image = tf.py_function(load_and_preprocess_image_tf, [file_path], tf.float32)
    return tf.reshape(image, (IMG_SIZE[1], IMG_SIZE[0], 3))

def tf_wrapper_load_and_preprocess_label(file_path):
    label = tf.py_function(load_and_preprocess_label, [file_path], tf.float32)
    return tf.reshape(label, [IMG_SIZE[1], IMG_SIZE[0], 3])

train_image_ds = train_image_ds.map(tf_wrapper_load_and_preprocess_image)
train_label_ds = train_label_ds.map(tf_wrapper_load_and_preprocess_label)
test_image_ds = test_image_ds.map(tf_wrapper_load_and_preprocess_image)
test_label_ds = test_label_ds.map(tf_wrapper_load_and_preprocess_label)

# Batch the datasets
BATCH_SIZE = 32
train_image_ds = train_image_ds.batch(BATCH_SIZE)
train_label_ds = train_label_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_label_ds = test_label_ds.batch(BATCH_SIZE)

# Zip the datasets
train_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds)).repeat()
val_ds = train_ds.take(1000 // BATCH_SIZE)
train_ds = train_ds.skip(1000 // BATCH_SIZE)

# Create a simple neural network for regression
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3)),  # Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Output layer for roof size prediction (regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Calculate steps per epoch
steps_per_epoch = len(os.listdir(TRAIN_IMAGE_DIR)) // BATCH_SIZE

# Train the model
model.fit(train_ds, epochs=10, validation_data=val_ds, validation_steps=len(os.listdir(TEST_IMAGE_DIR)) // BATCH_SIZE)
# Evaluate the model on the test dataset
test_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_loss = model.evaluate(test_ds, steps=len(os.listdir(TEST_IMAGE_DIR)) // BATCH_SIZE)
print(f"Test loss (MSE): {test_loss}")

# Make predictions for the test dataset
predictions = model.predict(test_image_ds)