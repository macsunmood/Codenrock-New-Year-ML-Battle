import os

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import (ModelCheckpoint, 
                                        EarlyStopping, 
                                        LearningRateScheduler)
# import keras_efficientnet_v2 as efn_v2
# from keras.utils import io_utils


### CONSTS
DATA_DIR = './data'
WEIGHTS_DIR = os.path.join(DATA_DIR, 'weight')  # model weights location

# WEIGHTS_DIR = '../data/weight/'

# VAL_SPLIT    = 0.002
# LR          = 0.001


### MODEL
def scheduler(epoch, lr, start=10):
    '''Defines scheduler for LearningRateScheduler'''
    if epoch < start:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def recreate_callbacks(es_patience=10, use_scheduler=True, scheduler_start=10, 
                       checkpoint_name='best_model.h5', best_only=True):
    '''Makes a list with new instances of each tensorflow callback'''
    
    callbacks_list = [ModelCheckpoint(os.path.join(WEIGHTS_DIR, checkpoint_name), 
                                      monitor='val_f1_score', 
                                      mode='max', 
                                      save_best_only=best_only, 
                                      save_weights_only=True, 
                                      verbose=0), 
                      EarlyStopping(monitor='val_f1_score', 
                                    patience=es_patience, 
                                    restore_best_weights=True)]
    if use_scheduler:
        callbacks_list.append(
            LearningRateScheduler(lambda ep, lr: 
                                  scheduler(ep, lr, scheduler_start), 
                                  verbose=1)
        )
    return callbacks_list


def pretrained_assemble(base, head, name=None):
    output = base.output

    for layer in head.layers:
        output = layer(output)

    return Model(inputs=base.input, 
                 outputs=output, 
                 name=name)


def classification_model(input_shape, num_classes=3):
    # base_model = efn_v2.EfficientNetV2S(num_classes=0, 
    #                                     pretrained='imagenet', 
    #                                     include_preprocessing=True, 
    #                                     input_shape=input_shape
    #                                     )

    # murl = 'https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-imagenet.h5'
    # base_model = tf.keras.utils.get_file('efficientnetv2-s-imagenet.h5', murl, 
    #                                      cache_subdir="models/efficientnetv2"
    #                                      )

    base_model = load_model(f'{WEIGHTS_DIR}/efficientnetv2-s-imagenet.h5')
    base_model = Model(inputs=base_model.input, 
                       outputs=base_model.layers[-4].output)

    head = Sequential([
        L.GlobalAveragePooling2D(), 
        L.Dense(1792, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu'), 
        L.Dense(896, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu'), 
        L.Dense(448, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu'), 
        L.Dense(224, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu'), 
        L.Dense(num_classes, activation='softmax')
    ])

    name = f'{base_model.name}_{input_shape[0]}x{input_shape[1]}'
    return pretrained_assemble(base_model, head, name)


def make_model(img_size):
    model = classification_model(input_shape=(img_size, img_size, 3))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True), 
                  metrics=[F1Score(num_classes=3, average='weighted')])

    return model


def train_model(model, train_gen, val_gen, img_size, batch_size, epochs):
    model.fit(train_gen, 
              validation_data=val_gen, 
              batch_size=batch_size, 
              epochs=epochs, 
              callbacks=recreate_callbacks(es_patience=10, scheduler_start=10), 
              verbose=1)

    model.load_weights(f'{WEIGHTS_DIR}/best_model.h5')
    return model


if __name__ == '__main__':
    print('train.py: OK')
