import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input image size
IMG_SIZE = (462, 300)

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
def load_and_preprocess_image(file_path):
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return image
def load_and_preprocess_label(file_path):
    label = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, IMG_SIZE)
    label = np.expand_dims(label, axis=-1)
    label = label / 255.0
    return label

train_image_ds = train_image_ds.map(load_and_preprocess_image)
train_label_ds = train_label_ds.map(load_and_preprocess_label)
test_image_ds = test_image_ds.map(load_and_preprocess_image)
test_label_ds = test_label_ds.map(load_and_preprocess_label)

# Split the training data into training and validation sets
train_image_ds = train_image_ds.shuffle(buffer_size=1000)
train_label_ds = train_label_ds.shuffle(buffer_size=1000)
train_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
val_ds = train_ds.take(1000)
train_ds = train_ds.skip(1000)

# Create a simple neural network for regression
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Input layer
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

# Train the model
model.fit(train_ds, epochs=10, validation_data=val_ds)

# Evaluate the model on the test dataset
test_loss = model.evaluate(test_ds)
print(f"Test loss (MSE): {test_loss}")

# Make predictions for the test dataset
predictions = model.predict(test_image_ds)

# Calculate the average accuracy