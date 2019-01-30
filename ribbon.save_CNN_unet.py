import nibabel as nib
import numpy as np
import glob

import json

from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, merge, Input, Lambda, Add, concatenate, Merge
from keras.optimizers import SGD,Adam
from scipy import stats
import itertools


def unet_model_2d(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    conv1 = Conv2D(32, conv_size, input_shape=input_shape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, conv_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(64, conv_size, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, conv_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(128, conv_size, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, conv_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(256, conv_size, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, conv_size, activation='relu', padding='same')(conv4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=512, image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(256, conv_size, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, conv_size, activation='relu', padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=256, image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(128, conv_size, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, conv_size, activation='relu', padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=128, image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(64, conv_size, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, conv_size, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(n_labels, (1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def unet_brown_2d_000(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(20, conv_size, activation='relu', padding='same')(conv1)
    # pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv2 = Conv2D(30, conv_size, activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv3 = Conv2D(40, conv_size, activation='relu', padding='same')(conv3)
    # pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    conv4 = Conv2D(50, conv_size, activation='relu', padding='same')(conv4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=50, image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(up5)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=40, image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(up6)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=30, image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(20, conv_size, activation='relu', padding='same')(up7)
    conv7 = Conv2D(20, conv_size, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(n_labels, (1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def unet_brown_2d_001(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same')(conv4)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same')(conv5)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same')(conv6)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv6], axis=3)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same')(up8)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv5], axis=3)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same')(up9)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, conv4], axis=3)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same')(up10)

    up11 = UpSampling2D(size=pool_size)(conv10)
    up11 = concatenate([up11, conv3], axis=3)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same')(up11)

    up12 = UpSampling2D(size=pool_size)(conv11)
    up12 = concatenate([up12, conv2], axis=3)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same')(up12)

    up13 = UpSampling2D(size=pool_size)(conv12)
    up13 = concatenate([up13, conv1], axis=3)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same')(up13)

    up14 = UpSampling2D(size=pool_size)(conv13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same')(up14)

    conv15 = Conv2D(n_labels, (1, 1))(conv14)
    act = Activation('sigmoid')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def unet_brown_2d_002(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same')(conv4)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same')(conv5)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same')(conv6)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv6], axis=3)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same')(up8)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv5], axis=3)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same')(up9)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, conv4], axis=3)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same')(up10)

    up11 = UpSampling2D(size=pool_size)(conv10)
    up11 = concatenate([up11, conv3], axis=3)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same')(up11)

    up12 = UpSampling2D(size=pool_size)(conv11)
    up12 = concatenate([up12, conv2], axis=3)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same')(up12)

    up13 = UpSampling2D(size=pool_size)(conv12)
    up13 = concatenate([up13, conv1], axis=3)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same')(up13)

    up14 = UpSampling2D(size=pool_size)(conv13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same')(up14)

    conv15 = Conv2D(n_labels, (1, 1))(conv14)
    act = Activation('softmax')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    return UpSampling2D(size=pool_size)



input_shape=(2560,2560,1)
# input_shape=(1200,1200,3)
# model = unet_model_2d(input_shape=input_shape, pool_size=(2, 2), n_labels=1)
# model = unet_brown_2d_001(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2)
model = unet_brown_2d_002(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2)
# print(model.summary)
json_string = model.to_json()
fn = "../model/NN_brown_unet_d2560_c5p2.n2soft.model.json"
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


