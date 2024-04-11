from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models
from emnist import extract_training_samples, extract_test_samples
from sklearn.model_selection import train_test_split
from input import blur, box_letters, filter_boxes, loadImg
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
    # model = models.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Dropout(rate=0.15),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(26, activation='softmax')  # 26 classes for letters
    # ])
    model = models.Sequential([
    layers.Conv2D(filters=473, kernel_size= (3, 3), activation=tf.nn.relu, input_shape=(28,28,1)),
    layers.AveragePooling2D((2, 2)),
    layers.Dropout(rate=0.15),
    layers.Conv2D(filters=238, kernel_size= (3, 3), padding='valid', activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.20),
    layers.Conv2D(filters=133, kernel_size= (3, 3), activation=tf.nn.relu),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.10),
    layers.Conv2D(filters=387, kernel_size= (3, 3), activation=tf.nn.relu),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.10),
    layers.Conv2D(filters=187, kernel_size= (5, 5), activation=tf.nn.elu),
    layers.Dropout(rate=0.50),
    layers.Dense(313, activation=tf.nn.relu),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.20),
    layers.Flatten(),
    layers.Dense(252, activation=tf.nn.elu),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.20),
    layers.Dense(37, activation=tf.nn.softmax)
    ])
    return model


model = build_model()

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Train model
def train_model(model, X_train, y_train, X_val, y_val):

    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))
    model.save_weights('model_weights.weights.h5')
    return history

# history = train_model(model, X_train, y_train, X_val, y_val)

# Evaluate model on test data
def test_model(X_test, y_test):
    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test Loss: ', test_loss)
    print('Test accuracy:', test_acc*100)

# test_model(X_test, y_test)

model.load_weights('model_weights.weights.h5')

test_model(X_test, y_test)

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
    prediction = np.array(predictions)
    predicted_class = np.argmax(prediction)
    predicted_letter = [class_to_letter_map[predicted_class]]
    print("Predicted class:", predicted_class)
    print("Predicted letter:", predicted_letter)

def classify_image(model):
    # Preprocess the image
    image = loadImg('images/test_boxing_2.jpg')
    _, boxes, _ = box_letters(image)
    boxes = filter_boxes(boxes)
    centers = mst.calculate_centers(boxes)
    dist, points_dict = mst.distance_matrix(centers)
    tree = mst.min_spanning_tree(dist, centers)

    newboxes, newtree = group_ij(boxes, tree)
    
    letters = extract_letters(image, newboxes)

    letters[0] = blur(letters[0])
    
    # Preprocess each letter image
    preprocessed_image = preprocess_image(letters[0])
    print('Letter shape ', preprocessed_image.shape)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title("First Letter")
    plt.axis('off')
    plt.show()
    # Convert the list of preprocessed images to a NumPy array
    input_data = np.array(preprocessed_image)
    input_data = np.expand_dims(input_data, axis=0)

    # print('Input data ', input_data.shape)

    # Predict the class probabilities
    predictions = model.predict(input_data)
    print(predictions)
    # Get the predicted class label
    print_letter(predictions)
    

classify_image(model)




