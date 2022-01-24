import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train import train_model


### CONSTS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PYTHONHASHSEED = 0

DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')    # train data location
TEST_DIR = os.path.join(DATA_DIR, 'test')      # where test images are mount for evaluation
OUT_DIR = os.path.join(DATA_DIR, 'out')        # dir for submission.csv
TRAIN_CSV = 'train.csv'

# IMG_SIZE = 672
# BATCH_SIZE = 12
IMG_SIZE     = 480 #384
BATCH_SIZE   = 8 #16
VAL_SPLIT    = 0.001


### DATASET
def rebuild_generators(data_df, target_size, batch_size, val_split, rescaling=True):
    train_df, val_df = train_test_split(
        data_df, 
        test_size=val_split,
        random_state=RANDOM_SEED,
        stratify=data_df['class_id'].values
    )

    rescale = 1. / 255 if rescaling else None

    train_datagen = ImageDataGenerator(
        rescale=rescale, 
        rotation_range=10, 
        brightness_range=[0.5, 1.5], 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        # horizontal_flip=True, 
    )
    val_datagen = ImageDataGenerator(rescale=rescale)
    # test_datagen = ImageDataGenerator(rescale=rescale)


    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory=DATA_DIR + 'train/', 
        x_col='image_name', 
        y_col='class_id', 
        target_size=target_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=True, 
        seed=RANDOM_SEED, 
        # subset='training'
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df, 
        directory=DATA_DIR + 'train/', 
        x_col='image_name', 
        y_col='class_id', 
        target_size=target_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=False, 
        # seed=RANDOM_SEED, 
        # subset='validation'
    )
    return train_gen, val_gen

### CLASS WEIGHTS
def get_class_weights(y):
    classes = np.unique(y)
    class_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=classes, 
        y=y
    )
    return {k: v for k, v in zip(classes, class_weights)}

### LOAD TO RAM
def to_RAM(gen):
    '''Loads sets of data to RAM'''
    X_ram = np.empty((0, *gen.image_shape), dtype=np.float32)
    y_ram = np.empty((0, len(gen.class_indices)), dtype=np.float32)

    gen.reset()
    for i in range(len(gen)):
        X, y = next(gen)
        X_ram = np.append(X_ram, X, axis=0)
        y_ram = np.append(y_ram, y, axis=0)

    print(f'[loaded] X: {len(X_ram)} samples; y: {len(y_ram)} labels')
    return [X_ram, y_ram]



if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV), sep='\t', dtype={'class_id': str})

    train_gen, val_gen = rebuild_generators(train_df, 
                                            target_size=(IMG_SIZE, IMG_SIZE), 
                                            batch_size=BATCH_SIZE, 
                                            val_split=VAL_SPLIT,
                                            rescaling=False)
    y_train = train_gen.labels
    y_val   = val_gen.labels
    y = np.concatenate((y_train, y_val))

    X_train_ram, y_train_ram = to_RAM(train_gen)
    X_val_ram, y_val_ram = to_RAM(val_gen)


    model = train_model(X_train_ram, y_train_ram, 
                        X_val_ram, y_val_ram, 
                        img_size=IMG_SIZE, 
                        batch_size=BATCH_SIZE)

    model.load_weights(f'{WORK_PATH}/best_model.h5')

    ### TEST
    test_df = pd.DataFrame(os.listdir(TEST_DIR), columns=['image_name'])

    datagen = ImageDataGenerator()
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df, 
        directory=TEST_DIR, 
        x_col='image_name',
        shuffle=False,
        batch_size=BATCH_SIZE,
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    y_pred = model.predict(test_gen)
    y_pred = np.argmax(y_pred, axis=1)
    print('Ok')


    ### SUBMISSION
    submission = test_df.copy()
    submission['class_id'] = y_pred
    submission.to_csv(os.path.join(OUT_DIR, 'submission.csv'), 
                      index=False, sep='\t')
    print('Ok')