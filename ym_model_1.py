#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, to_categorical
import glob
import random
import cv2
from random import shuffle
import voxel
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def dataload():
    img = np.load("imgs.npy")
    mask = np.load("masks.npy")
    
    # shuffle
    shuffle = np.arange(img.shape[0])
    np.random.shuffle(shuffle)
    img = img[shuffle]
    mask = mask[shuffle]

    img = np.expand_dims(img, axis=-1)
    mask = np.expand_dims(mask, axis=-1) 
#     mask = to_categorical(mask) 
    
    return img, mask 

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

# def minmaxScaler(data):

# def dice_coef(y_true, y_pred, smooth=1):
#     y_pred = K.argmax(y_pred, axis=-1)
# #     y_true = y_true[:,:,:,0]

#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     y_true_f = K.cast(y_true_f, 'float32')
#     y_pred_f = K.cast(y_pred_f, 'float32')

#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice = 0

    for index in range(numLabels):
        if index == 0: continue
        dice += dice_coef_each(y_true, y_pred, index)
        
    return dice / numLabels # taking average

def dice_coef_each(y_true, y_pred, label, smooth=1):
    y_true = K.cast(K.equal(y_true, label), 'float32')
    y_pred = K.cast(K.equal(K.argmax(y_pred, axis=-1), label), 'float32')
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_0(y_true, y_pred):
    return dice_coef_each(y_true, y_pred, 0)

def dice_coef_1(y_true, y_pred):
    return dice_coef_each(y_true, y_pred, 1)

def dice_coef_2(y_true, y_pred):
    return dice_coef_each(y_true, y_pred, 2)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = y_true[:,:,:,1]
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return tf.math.exp(1  - score) - 1.0
    # return 1. - score

def bce_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def build_unet(sz=(512,512,1)):
    x = Input(sz)
    inputs = x
  
    #down sampling 
    f = 8
    layers = []
  
    for i in range(0, 6):
        # kernel_initializer='he_norm' kernel의 값을 맞춰줄 수 있음. (초기화 설정) he_norm 앞 레이어의 평균과 표준편차를 맞춰서 정규화를 해준다.
        x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
        # bias는 BatchNormalization에서 조절
        x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal') (x)
        # BatchNormalization
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        layers.append(x)
        x = MaxPooling2D() (x)
        f = f*2
        ff2 = 64 
    
    #bottleneck 
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
    x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
    x = Dropout(0.25)(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
  
    #upsampling 
    for i in range(0, 5):
        ff2 = ff2//2
        f = f // 2 
        x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
        x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same', use_bias=False) (x)
        # BatchNormalization
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j -1 
    
    #classification 
    x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
    x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal') (x)
    x = Dropout(0.25)(x)
    outputs = Conv2D(3, 1, activation='softmax') (x)
    
    #model creation 
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', 
                  metrics = [dice_coef_0, dice_coef_1, dice_coef_2])
  
    return model
# model = build_unet()
# model.save('./model/model.h5')


# In[2]:


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # source, target = data_convert()

    source, target = dataload()
    print(source.shape, target.shape)
    
    model = build_unet()
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(filepath='./model/{epoch}_{val_loss:.4f}.h5',
                                      monitor='val_loss',
                                      save_best_only=True,
                                      verbose=1,
                                      mode='auto')
    model.fit(source, target, epochs=100, validation_split=0.2, batch_size=8, callbacks=[early_stopping, model_checkpoint])

