import os 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import and read in data, separate the data into training and dev data sets
mnist = tf.keras.datasets.mnist
(pixel_train, digit_train), (pixel_test, digit_test) = mnist.load_data()

# Scaling pixel values from data to be between 0-1
pixel_train = tf.keras.utils.normalize(pixel_train, axis=1)
pixel_test = tf.keras.utils.normalize(pixel_test, axis=1)

# Create the learning model and add neural network layers that will be applied to activation functions 
learning_model = tf.keras.models.Sequential()
learning_model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
learning_model.add(tf.keras.layers.Dense(128,activation='relu'))
learning_model.add(tf.keras.layers.Dense(128,activation='relu'))
learning_model.add(tf.keras.layers.Dense(10,activation='softmax'))

# Create a compile function and choose the optimizer
learning_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now train the model using the data for an initial train test
learning_model.fit(pixel_train, digit_train, epochs=5)

# Evualate the model to see how well it performs with train data cross-checked with dev data
loss, accuracy = learning_model.evaluate(pixel_test, digit_test)

# Use a file of digit image data to test the learning model
image_number=0
while os.path.isfile(f'digittest/{image_number}.png'):
    try:
        image = cv2.imread(f'digittest/{image_number}.png')[:,:,0]
        image = np.invert(np.array([image]))
        prediction = learning_model.predict(image)
        print(f"Digit prediction: {np.argmax(prediction)}")
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error no images found!")
    finally:
        image_number+=1



