import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('./letnet5.h5')

# Check its architecture
#new_model.summary()
def prep(image):
    IMG_SIZE = 28
    img_array = cv2.imread(image)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

prediction = new_model.predict([prep('./Capture.png')])

names = ['1', '2', '3','4','5','6','7','8','9','10']
values = prediction[0]

plt.figure()
plt.bar(names, values)
plt.show()