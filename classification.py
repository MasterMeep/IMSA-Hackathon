import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from PIL import Image


#loading model
model = tf.keras.models.load_model('./trainedModel')

#general parameters
batch_size = 32
img_height = 180
img_width = 180

#classifications
class_names = ['NORMAL', 'PNEUMONIA']

#classify image from local path
def classify(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img.save('image.jpg', 'jpeg')

    img = tf.keras.utils.load_img(
        './image.jpg', target_size=(img_height, img_width)
    )
    img_array  = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return(class_names[np.argmax(score)])

