from tkinter import Image
import numpy as np
import emnist
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, AveragePooling2D

# TODO: Create dataset load function
# TODO: Create train function
# TODO: Create test function
# TODO: Create a function to predict a single image


def load_dataset():
    #Load the EMNIST dataset
    ds_train, ds_test, ds_info = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

def load_ds():

    # Load EMNIST dataset
    emnist_train = emnist.extract_training_samples('letters')
    emnist_test = emnist.extract_test_samples('letters')

    # Extract images and labels
    X_train, y_train = emnist_train
    X_test, y_test = emnist_test

    # Reshape images for TensorFlow model input
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

    y_train = tf.cast(y_train, dtype=tf.int32)
    y_test = tf.cast(y_test, dtype=tf.int32)

    # Define batch size
    batch_size = 128

    # Create TensorFlow Dataset objects for training and testing
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=X_train.shape[0])  # Shuffle the dataset
    ds_train = ds_train.batch(batch_size)  # Batch the dataset
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch data for efficiency

    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.batch(batch_size)  # Batch the dataset
    ds_test = ds_test.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch data for efficiency
    
    return ds_train, ds_test

def build_model(input_shape=(28, 28, 1)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=473, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Conv2D(filters=238, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.20),
        tf.keras.layers.Conv2D(filters=133, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.10),
        tf.keras.layers.Conv2D(filters=387, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.10),
        tf.keras.layers.Conv2D(filters=187, kernel_size=(5, 5), activation=tf.nn.elu),
        tf.keras.layers.Dropout(rate=0.50),
        tf.keras.layers.Dense(313, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.20),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(252, activation=tf.nn.elu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.20),
        tf.keras.layers.Dense(37)
    ])

    return model

def train_model(model, ds_train, ds_test, epochs=10):
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test
    )
    
    return history

def test_model(model, ds_test):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')

def classify_image(model, image):
    # Preprocess the image
    image_path = "path_to_your_image.jpg"
    image = Image.open(image_path)
    image = tf.image.resize(image, (28, 28))  # Resize the image to match the input shape of the model
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.cast(image, tf.float32) / 255.  # Normalize the image

    # Predict the class probabilities
    predictions = model.predict(image)
    
    # Get the predicted class label
    predicted_class = tf.argmax(predictions[0]).numpy()
    
    return predicted_class

# Load dataset
#ds_train, ds_test = load_dataset()
ds_train, ds_test = load_ds()

# Build the model
model = build_model()

# Train the model
history = train_model(model, ds_train, ds_test, epochs=10)


# Test the model
test_model(model, ds_test)
