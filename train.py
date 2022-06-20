import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import (ModelCheckpoint, 
                                        EarlyStopping, 
                                        LearningRateScheduler)
import keras_efficientnet_v2 as efn_v2
# from keras.utils import io_utils


### MODEL
def scheduler(epoch, lr, start=10):
    '''Defines scheduler for LearningRateScheduler'''
    if epoch < start:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def recreate_callbacks(es_patience=10, use_scheduler=True, scheduler_start=10, 
                       checkpoint_path='best_model.h5', best_only=True):
    '''Makes a list with new instances of each tensorflow callback'''
    
    callbacks_list = [ModelCheckpoint(filepath=checkpoint_path, 
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
    '''Assembles a pretrained base model with a new head'''
    output = base.output

    for layer in head.layers:
        output = layer(output)

    return Model(inputs=base.input, 
                 outputs=output, 
                 name=name)


def classification_model(input_shape, num_classes):
    '''Classification model for Santa contest'''
    base_model = efn_v2.EfficientNetV2S(num_classes=0,
                                        pretrained='imagenet', 
                                        include_preprocessing=True, 
                                        input_shape=input_shape
                                        )

    head = Sequential([
        L.GlobalAveragePooling2D(), 
        L.Dense(1792, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu', name='activation_'), 
        L.Dense(896, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu', name='activation_1_'), 
        L.Dense(448, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu', name='activation_2_'), 
        L.Dense(224, use_bias=True, kernel_regularizer='l2'), 
        L.Dropout(0.5), 
        L.Activation('selu', name='activation_3_'), 
        L.Dense(num_classes, activation='softmax')
    ])

    name = f'{base_model.name}_{input_shape[0]}x{input_shape[1]}'
    return pretrained_assemble(base_model, head, name)


def make_model(img_size, learning_rate, num_classes=3):
    '''Makes a compiled model'''
    model = classification_model(input_shape=(img_size, img_size, 3), 
                                 num_classes=num_classes)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=learning_rate, amsgrad=True), 
                  metrics=[F1Score(num_classes=3, average='weighted')])

    return model


def train_model(model, train_gen, val_gen, img_size, batch_size, epochs, weights_dir):
    '''Training routing for model'''
    model.fit(train_gen, 
              validation_data=val_gen, 
              batch_size=batch_size, 
              epochs=epochs, 
              callbacks=recreate_callbacks(
                  es_patience=10, 
                  scheduler_start=10, 
                  checkpoint_path=f'{weights_dir}/best_model.h5'
              ), 
              verbose=1)

    model.load_weights(f'{weights_dir}/best_model.h5')
    return model


if __name__ == '__main__':
    print('train.py: OK')
