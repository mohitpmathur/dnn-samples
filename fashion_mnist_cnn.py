"""
Create a CNN model to classify fashion MNIST dataset images
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print("Fashion MNIST dataset loaded successfully!!")

print(f"Shape of train_images: {train_images.shape}")

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1)),
                                    MaxPooling2D(2, 2),
                                    Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
                                    MaxPooling2D((2, 2)),
                                    # Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
                                    # MaxPooling2D((2,2)),
                                    Flatten(),
                                    Dense(128, activation='relu'),
                                    Dense(10, activation='softmax')])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model Summary:", model.summary())

es = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(train_images,
                    train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=30,
                    batch_size=256,
                    callbacks=[es])

predictions = model.predict(test_images)

print(f"First prediction: {np.argmax(predictions[15])}")
print(f"First label: {test_labels[15]}")
