import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, concatenate
from data_loader import load_data

def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1, 1), activation='relu', name=None):
    """Helper function to perform convolution followed by batch normalization"""
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation, name=name)(x)
    x = BatchNormalization()(x)
    return x
def inception_module(x, params):
    """
    Implementation of an Inception module.
    Args:
        x: input tensor
        params: list of parameters defining the module
    Returns:
        output tensor
    """
    tower_1 = conv2d_bn(x, params[0], (1, 1))

    tower_2 = conv2d_bn(x, params[1], (1, 1))
    tower_2 = conv2d_bn(tower_2, params[2], (3, 3))

    tower_3 = conv2d_bn(x, params[3], (1, 1))
    tower_3 = conv2d_bn(tower_3, params[4], (5, 5))

    tower_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_4 = conv2d_bn(tower_4, params[5], (1, 1))

    output = concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
    return output

def stage_1(input_img):
        """Implementation of the first stage of the Inception v1 model"""
        x = conv2d_bn(input_img, 64, (7, 7), strides=(2, 2))
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        return x  

def stage_2(x):
    """Implementation of the second stage of the Inception v1 model"""
    params = [[64, 96, 128, 16, 32, 32],
              [128, 128, 192, 32, 96, 64]]

    for p in params:
        x = inception_module(x, p)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x

   

def stage_3(x):
    """Implementation of the third stage of the Inception v1 model"""
    params = [[192, 96, 208, 16, 48, 64],
              [160, 112, 224, 24, 64, 64],
              [128, 128, 256, 24, 64, 64]]

    for p in params:
        x = inception_module(x, p)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x

def stage_4(x):
    """Implementation of the fourth stage of the Inception v1 model"""
    params = [[192, 96, 208, 16, 48, 64],
              [160, 112, 224, 24, 64, 64],
              [128, 128, 256, 24, 64, 64],
              [112, 144, 288, 32, 64, 64],
              [256, 160, 320, 32, 128, 128]]

    for i in range(len(params)):
        if i == len(params)-1:
            x = inception_module(x, params[i])
        else:
            x = inception_module(x, params[i])
            x = inception_module(x, params[i])

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x


def inception_v1(input_shape, num_classes):
    """Implementation of the Inception v1 architecture"""
    input_img = Input(shape=input_shape)

    x = stage_1(input_img)
    x = stage_2(x)
    x = stage_3(x)
    x = stage_4(x)

    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer with dropout regularization
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Add a softmax layer for the output
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    return model

