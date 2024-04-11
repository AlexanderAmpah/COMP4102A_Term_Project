import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import layers, models
from emnist import extract_training_samples, extract_test_samples
from sklearn.model_selection import train_test_split


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

    train = (X_train, y_train)
    validation = (X_val, y_val)
    test = (X_test, y_test)

    return train, validation, test


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

    # Compile model

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


def load_model(model, filename='model_weights.weights.h5'):    
    model.load_weights(filename)


def save_model(model, filename='model_weights.weights.h5'):
    model.save_weights(filename)


def train_model(model, train, validation):
    X_train, y_train = train
    X_val, y_val = validation
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val))

    return history


def test_model(model, test):
    X_test, y_test = test
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test Loss: ', test_loss)
    print('Test accuracy:', test_acc * 100)

    return test_loss, test_acc


def preprocess_image(image):
    # Ensure the image has 3 or 4 dimensions
    # Add a channel dimension if the image has 2 dimensions

    _, threshold = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    if len(image.shape) == 2:
        tmp = threshold[:, :, None] 

    # Resize the image to match the input size required by the model
    # Normalize pixel values

    # Emnist letters do not connect with image edges (at least 2 pixes between edge and image)

    paddings = tf.constant([
        [2, 2], [2, 2]
    ])
    
    resized_image = tf.image.resize_with_pad(tmp, 24, 24)
    resized_image = resized_image[:, :, 0]

    resized_image = tf.pad(resized_image, paddings, mode='CONSTANT')

    resized_image = np.array(resized_image)
    resized_image = cv.GaussianBlur(resized_image, (3, 3), 1)

    return resized_image


def classify(model, letter):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
               'h', 'i', 'j', 'k', 'l', 'm', 'n', 
               'o', 'p', 'q', 'r', 's', 't', 'u', 
               'v', 'w', 'x', 'y', 'z']

    # Preprocess each letter image

    preprocessed_letter = preprocess_image(letter)
    # preprocessed_image = tf.where(preprocessed_image >= 0.6, 1.0, 0.0)

    # print('Letter shape ', preprocessed_image.shape)
    # Convert the list of preprocessed images to a NumPy array

    

    input_data = np.array(preprocessed_letter)
    input_data = np.expand_dims(input_data, axis=0)

    # Predict the class probabilities
    predictions = model.predict(input_data)
    prediction = np.array(predictions)

    predicted_class = np.argmax(prediction)
    predicted_letter = letters[predicted_class - 1]     # One based indexing. Subtract 1 to get 0 based indexing

    return predicted_letter, predicted_class 