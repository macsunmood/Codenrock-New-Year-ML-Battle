import os
import subprocess
import numpy as np

from dataset import ImagesDataset
from train import make_model, train_model

from tensorflow.keras.models import load_model


### CONSTS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PYTHONHASHSEED = 0

DATA_DIR = './data'
# DATA_DIR = '../data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')     # train data location
WEIGHTS_DIR = os.path.join(DATA_DIR, 'weight')  # model weights location
TEST_DIR = os.path.join(DATA_DIR, 'test')       # where test images are mount for evaluation
OUT_DIR = os.path.join(DATA_DIR, 'out')         # dir for submission.csv
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

VAL_SPLIT  = 0.01
IMG_SIZE   = 384
# IMG_SIZE   = 8
LR         = 0.0001
BATCH_SIZE = 8
# BATCH_SIZE = 64
NUM_EPOCHS = 20
# NUM_EPOCHS = 1


if __name__ == '__main__':
    try:
        subprocess.call('nvidia-smi')
    except:
        pass

    dataset = ImagesDataset(
        data_dir=TRAIN_DIR, 
        data_csv=TRAIN_CSV, 
        val_split=VAL_SPLIT, 
        rescale=None, 
        image_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE, 
        random_state=RANDOM_SEED
    )

    dataset.build_generators()

    # train_gen, val_gen = build_generators(
    #     augment_image, 
    # )


    ### ---TRAINING PART: START    
    # model = make_model(
    #     img_size=IMG_SIZE, 
    #     learning_rate=LR, 
    #     num_classes=dataset.num_classes
    # )
    
    # model = train_model(
    #     model, 
    #     dataset.train_gen, 
    #     dataset.val_gen, 
    #     img_size=IMG_SIZE, 
    #     batch_size=BATCH_SIZE, 
    #     epochs=NUM_EPOCHS,
    #     weights_dir=WEIGHTS_DIR
    # )

    # model.save(f'{WEIGHTS_DIR}/{model.name}.h5')
    ### ---TRAINING PART: END


    # Download pretrained model with weights
    # import gdown
    # WEIGHTS_FILE_ID = '1Id9r8YSq_XCKeF0hz020hvdhQd6jETOE'
    # gdown.download(id=WEIGHTS_FILE_ID, output=f'{WEIGHTS_DIR}/model.h5', quiet=False)

    # Load pretrained model for inference
    model = load_model(f'{WEIGHTS_DIR}/model.h5')

    ### TEST PREDICTION
    dataset.build_test_generator(TEST_DIR)

    y_pred = model.predict(dataset.test_gen)
    y_pred = np.argmax(y_pred, axis=1)
    print('Prediction [OK]')

    ### SUBMISSION
    submission = dataset.test_df.copy()
    submission['class_id'] = y_pred
    submission.to_csv(os.path.join(OUT_DIR, 'submission.csv'), 
                      index=False, sep='\t')
    print('Submission [OK]')
