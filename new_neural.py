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
from input import box_letters, filter_boxes, loadImg, plotBoxes
from letters import extract_letters, group_ij
import mst

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
        layers.Dropout(rate=0.15),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(26, activation='softmax')  # 26 classes for letters
    ])
    return model


model = build_model()

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Train model
def train_model(model, X_train, y_train, X_val, y_val):

    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))
    model.save_weights('model_weights.weights.h5')
    return history

history = train_model(model, X_train, y_train, X_val, y_val)

# Evaluate model on test data
def test_model(X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test Loss: ', test_loss)
    print('Test accuracy:', test_acc*100)

test_model(X_test, y_test)

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

def print_letter(predictions):
    class_to_letter_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }

    predicted_class = np.argmax(predictions, axis=1)
    predicted_letter = [class_to_letter_map[class_idx] for class_idx in predicted_class]
    print("Predicted class:", predicted_class)
    print("Predicted letter:", predicted_letter)

def classify_image(model):
    # Preprocess the image
    image = loadImg('images/test_boxing.jpg')
    _, boxes, _ = box_letters(image)
    boxes = filter_boxes(boxes)
    centers = mst.calculate_centers(boxes)
    dist, points_dict = mst.distance_matrix(centers)
    tree = mst.min_spanning_tree(dist, centers)

    newboxes, newtree = group_ij(boxes, tree)
    
    letters = extract_letters(image, newboxes)

    plt.imshow(letters[0], cmap='gray')
    plt.title("First Letter")
    plt.axis('off')
    plt.show()

    # Preprocess each letter image
    preprocessed_image = preprocess_image(letters[0])
    # Convert the list of preprocessed images to a NumPy array
    input_data = np.array(preprocessed_image)
    input_data = np.expand_dims(input_data, axis=0)

    # Predict the class probabilities
    predictions = model.predict(input_data)
    
    # Get the predicted class label
    print_letter(predictions)
    

classify_image(model)




