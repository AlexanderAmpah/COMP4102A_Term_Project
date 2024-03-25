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


# Load the EMNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
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


# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=473, kernel_size= (3, 3), activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Conv2D(filters=238, kernel_size= (3, 3), padding='valid', activation=tf.nn.leaky_relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.20),
    tf.keras.layers.Conv2D(filters=133, kernel_size= (3, 3), activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.10),
    tf.keras.layers.Conv2D(filters=387, kernel_size= (3, 3), activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.10),
    tf.keras.layers.Conv2D(filters=187, kernel_size= (5, 5), activation=tf.nn.elu),
    tf.keras.layers.Dropout(rate=0.50),
    tf.keras.layers.Dense(313, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(252, activation=tf.nn.elu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.20),
    tf.keras.layers.Dense(37, activation=tf.nn.softmax)

])
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=tf.keras.optimizers.RMSprop(),
#     metrics=['accuracy'],
# )

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(ds_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
