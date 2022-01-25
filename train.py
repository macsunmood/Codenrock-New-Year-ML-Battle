import os

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as L
from tensorflow.keras import regularizers, optimizers

from tensorflow.keras.callbacks import (ModelCheckpoint, 
                                        EarlyStopping, 
                                        LearningRateScheduler,
                                        ReduceLROnPlateau)
import keras_efficientnet_v2 as efn_v2


### CONSTS
DATA_DIR = './data'
WORK_DIR = os.path.join(DATA_DIR, 'weight')    # place to keep models weights at

# LR          = 0.001
# DROPOUT     = 0.25
# NUM_EPOCHS  = 20


### CALLBACKS
def scheduler(epoch, lr, start=10):
    '''Defines scheduler for LearningRateScheduler'''
    if epoch < start:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def recreate_callbacks(es_patience=10, use_scheduler=True, scheduler_start=10, 
                       checkpoint_name='best_model.h5', best_only=True):
    '''Makes a list with new instances of each tensorflow callback'''
    
    callbacks_list = [ModelCheckpoint(os.path.join(WORK_DIR, checkpoint_name), 
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


### MODEL
def pretrained_assemble(base, head, name=None):
    output = base.output
    
    for l in head.layers:
        output = l(output)

    return Model(inputs=base.input, 
                 outputs=output, 
                 name=name)

def classification_model(input_shape, num_classes=3):

    base_model = efn_v2.EfficientNetV2S(num_classes=0, 
                                        pretrained='imagenet', 
                                        include_preprocessing=True, 
                                        input_shape=input_shape
                                        )

    head = Sequential([
        L.GlobalAveragePooling2D(), 
        L.Dense(256, use_bias=False, kernel_regularizer='l2'), 
        L.BatchNormalization(axis=1), 
        L.Activation('relu'), 
        L.Dropout(0.35), #DROPOUT
        L.Dense(num_classes, activation='softmax')
    ])
    
    name = f'EfficientNetV2_{input_shape[0]}x{input_shape[1]}'
    return pretrained_assemble(base_model, head, name)

def train_model(X_train, y_train, X_val, y_val, img_size, batch_size):
    model = classification_model(input_shape=(img_size, img_size, 3))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True), 
                  metrics=[F1Score(num_classes=3, average='weighted')])

    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              batch_size=batch_size, 
              epochs=40, #NUM_EPOCHS, 
              callbacks=recreate_callbacks(es_patience=10, scheduler_start=10), 
              verbose=1)

    model.load_weights(f'{WORK_DIR}/best_model.h5')
    return model


if __name__ == '__main__':
    print('Ok')