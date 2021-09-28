"""
Create a NN model to classify fashion MNIST dataset images
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print("Fashion MNIST dataset loaded successfully!!")

print(f"Shape of train_images: {train_images.shape}")

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    Dense(128, activation='relu'),
                                    Dense(64, activation='relu'),
                                    Dense(10, activation='softmax')])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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
