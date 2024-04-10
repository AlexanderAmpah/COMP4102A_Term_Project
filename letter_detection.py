from tkinter import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, AveragePooling2D
from emnist import extract_training_samples, extract_test_samples
import sklearn
from sklearn.model_selection import train_test_split
from input import box_letters, loadImg
from letters import extract_letters

def load_dataset():
    # Load EMNIST letters dataset
    X_train, y_train = extract_training_samples('letters')
    X_test, y_test = extract_test_samples('letters')

    # Filter out labels >= 26
    valid_indices_train = np.where(y_train < 26)
    valid_indices_test = np.where(y_test < 26)
    X_train, y_train = X_train[valid_indices_train], y_train[valid_indices_train]
    X_test, y_test = X_test[valid_indices_test], y_test[valid_indices_test]

    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape data to fit CNN input shape
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model():
    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(26, activation='softmax')  # 26 classes for letters
    ])
    return model


model = build_model()

# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Train model
def train_model(model, X_train, y_train, X_val, y_val):

    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
    model.save_weights('model_weights.weights.h5')
    return history

# history = train_model(model, X_train, y_train, X_val, y_val)

# Evaluate model on test data
def test_model(X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test Loss: ', test_loss)
    print('Test accuracy:', test_acc*100)

# test_model(X_test, y_test)

model.load_weights('model_weights.weights.h5')

def preprocess_image(image):
    # Ensure the image has 3 or 4 dimensions
    if len(image.shape) == 2:
        # Add a channel dimension if the image has 2 dimensions
        image = np.expand_dims(image, axis=-1)
    # Resize the image to match the input size required by the model
    resized_image = tf.image.resize(image, (28, 28))
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    return normalized_image

def classify(letter, model):
    # Preprocess the image   
    
    preprocessed= preprocess_image(letter)
    # Convert the list of preprocessed images to a NumPy array
    input_data = np.array(preprocessed)

    # Predict the class probabilities
    predictions = model.predict(input_data)
    
    # Get the predicted class label
    # predicted_class = tf.argmax(predictions[0]).numpy()
    predicted_class = tf.argmax(predictions, axis=-1)

    print('Predicted class: ', predicted_class)
    
    return predicted_class

classify_image(model)



def print_word(class_labels):
    # Map class labels to letters (assuming class labels correspond to letter indices)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Convert class labels to letters
    word = ''.join([letters[label] for label in class_labels])
    
    print("Word formed by the letters:", word)