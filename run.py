
import os
import subprocess
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from dataset import augment_image, rebuild_generators
from train import make_model, train_model


from tensorflow.keras.models import load_model


### CONSTS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PYTHONHASHSEED = 0

DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')     # train data location
WEIGHTS_DIR = os.path.join(DATA_DIR, 'weight')  # model weights location
TEST_DIR = os.path.join(DATA_DIR, 'test')       # where test images are mount for evaluation
OUT_DIR = os.path.join(DATA_DIR, 'out')         # dir for submission.csv
TRAIN_CSV = 'train.csv'

VAL_SPLIT  = 0.002
IMG_SIZE   = 384
BATCH_SIZE = 16
NUM_EPOCHS = 30


if __name__ == "__main__":
    subprocess.call('nvidia-smi')

    # model = make_model(IMG_SIZE)
    
    # model.load_weights(f'{WEIGHTS_DIR}/best_model.h5')

    # train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV), 
    #                        sep='\t', dtype={'class_id': str})

    # train_gen, val_gen = rebuild_generators(
    #     train_df, 
    #     TRAIN_DIR, 
    #     augment_image, 
    #     target_size=(IMG_SIZE, IMG_SIZE), 
    #     batch_size=BATCH_SIZE, 
    #     val_split=VAL_SPLIT, 
    #     random_state=RANDOM_SEED, 
    #     rescaling=False
    # )

    # model = train_model(
    #     model, 
    #     train_gen, 
    #     val_gen, 
    #     img_size=IMG_SIZE, 
    #     batch_size=BATCH_SIZE, 
    #     epochs=NUM_EPOCHS 
    # )
    
    # model.save(f'{WEIGHTS_DIR}/{model.name}.h5')

    
    model = load_model(f'{WEIGHTS_DIR}/model.h5')

    ### TEST PREDICTION
    # TEST_DIR = '../data/test/'
    test_df = pd.DataFrame(os.listdir(TEST_DIR), columns=['image_name'])

    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df, 
        directory=TEST_DIR, 
        x_col='image_name',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False
    )

    y_pred = model.predict(test_gen)
    y_pred = np.argmax(y_pred, axis=1)
    print('Prediction [OK]')

    ### SUBMISSION
    submission = test_df.copy()
    submission['class_id'] = y_pred
    submission.to_csv(os.path.join(OUT_DIR, 'submission.csv'), 
                      index=False, sep='\t')
    print('Submission [OK]')
