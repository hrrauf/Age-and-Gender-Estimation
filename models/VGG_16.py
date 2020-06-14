# This code is imported from the following project: https://github.com/asmith26/wide_resnets_keras

import logging
import sys
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D, Concatenate, Lambda
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class VGG_16:
    def __init__(self, num_classes):
        k=10
        self.num_classes=num_classes
     



    def __call__(self,pretrained=True):
        self.pretrained= pretrained
#         logging.debug("Creating model...")

       
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
 
        model.add(Conv2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        if self.pretrained==True:
        
            model.load_weights('./trained_models/vgg_face_weights.h5')
        
        for layer in model.layers[:-7]:
            layer.trainable = False



        base_model_output = Convolution2D(self.num_classes, (1, 1), name='predictions')(model.layers[-4].output)

        base_model_output = Flatten()(base_model_output)

        base_model_output = Activation('softmax')(base_model_output)


        model = Model(inputs=model.input, outputs=base_model_output)

        return model


def main():
    model = VGG_16()()
   

if __name__ == '__main__':
    main()
