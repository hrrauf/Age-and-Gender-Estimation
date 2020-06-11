from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D, Concatenate, Lambda
from keras.models import Model, Sequential
from keras import metrics
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import torch
from contextlib import contextmanager
from keras.preprocessing.image import load_img, img_to_array
from VGG import gen_vgg_model
graph = tf.get_default_graph()

class ImageClassifier():
    print("***************************")
    print("Initializing classifier")
    
    def __init__(self,modelName="vgg"):
        
        self.modelName = modelName
        
        if modelName=='vgg':
            age_model=gen_vgg_model(101)
            age_model.load_weights('./models/age_model_weights.h5')
            gender_model=gen_vgg_model(2)
            gender_model.load_weights('./models/gender_model_weights.h5')
            self.age_model = age_model
            self.gender_model = gender_model
            self.img_size=224
            
        elif modelName=='wide_resnet':
            model= WideResNet(64, depth=16, k=8)()
            model.load_weights('./models/weights.28-3.73.hdf5')
            self.img_size=64
            self.model = model
        
        print('Initialization complete')
        print("***************************")
            
    def draw_label(self,image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.8, thickness=2):

        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        img = cv2.putText(image, label, point, font, font_scale, (255, 0, 255), 3)
        return img
        
    def classify_image(self,image):
        
        detector = dlib.get_frontal_face_detector()
        margin=0.4
        
        img = cv2.resize(image, (self.img_size,self.img_size))
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = np.shape(input_img)
        ih, iw, _ = np.shape(image)

        detected = detector(image, 1)
        faces = np.empty((len(detected), self.img_size, self.img_size, 3))
        
        for i, d in enumerate(detected):
            d=detected[i]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), iw - 1)
            yw2 = min(int(y2 + margin * h), ih - 1)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

            faces[i, :, :, :] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))
            if self.modelName=='vgg':
                with graph.as_default():
                    results_age = self.age_model.predict(np.expand_dims(faces[i,:,:,:],axis=0))
                    predicted_genders=self.gender_model.predict(np.expand_dims(faces[i,:,:,:],axis=0))
                
            elif self.modelName=='wide_resnet':
                with graph.as_default():
                    results = self.model.predict(np.expand_dims(faces[i,:,:,:],axis=0))
                results_age=results[1]
                predicted_genders=results[0]

            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results_age[0].dot(ages).flatten()

            label = "{}, {}".format(int(predicted_ages),"M" if np.argmax(predicted_genders)==1 else "F")
            image = self.draw_label(image,(d.left(), d.top()), label)
            
        return image