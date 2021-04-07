import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
import PIL
import PIL.Image
from keras.models import load_model
import h5py
import pathlib
import cv2
import tensorflow.keras as keras

import os
import zipfile


import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

def create_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer=Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
model.load_weights('F:\mask detection\checkpoints')

# model = model.load_weights(latest)

print(model)


import numpy as np

from keras.preprocessing import image



# # predicting images
# path = 'F:\mask detection/320.jpg'
# img = image.load_img(path, target_size=(150, 150))
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# images = np.vstack([x])
#
# classes = model.predict(images, batch_size=20)
#
# print(classes[0])
#
# if classes[0] > 0:
#     print(" is without")
#     print(classes[0])
#
# else:
#     print(" is a with mask")
#     print(classes[0])
#
#




import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        #scaleFactor=1.3,
        #minNeighbors=3
        minSize=(200, 200)
    )

    img_with_detections = np.copy(frame)
    if(len(faces)>0):
        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            immg = cv2.resize(roi_color , (150,150))

            xx = image.img_to_array(immg)
            xx = np.expand_dims(xx, axis=0)
            images = np.vstack([xx])

            classes = model.predict(images, batch_size=20)

            print(classes[0])

            if classes[0] > 0:
                print(" is with")
                print(classes[0])
                cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img_with_detections, 'with mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            else:
                print(" is a without ")
                cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img_with_detections, 'without mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                print(classes[0])


    cv2.imshow('frame',img_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAll

