import os
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### CONSTS
RANDOM_SEED = 42

DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')     # train data location


# Create an image augmentation object
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5), 

    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                       rotate_limit=15, p=0.5), 

    A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.25, 
                                        contrast_limit=0.25), 
             A.RandomBrightnessContrast(brightness_limit=0.1, 
                                        contrast_limit=0.1)], p=0.2),
    
    A.RandomGamma(p=0.2), 

    A.GaussNoise(p=0.15), 
    
    A.Blur(blur_limit=3, p=0.2)
])


# Augmentation of images function
def augment_image(image):
    aug_img = np.array(image)
    aug_img = augmentation(image=image)['image']
    return aug_img


# Data Generators
def rebuild_generators(df, augment_func, train_dir, target_size, batch_size, val_split, random_state, rescaling=True):
    train_df, val_df = train_test_split(
        df, 
        test_size=val_split,
        random_state=random_state,
        stratify=df['class_id'].values
    )

    rescale = 1. / 255 if rescaling else None

    train_datagen = ImageDataGenerator(preprocessing_function=augment_func, 
                                       rescale=rescale)
    val_datagen = ImageDataGenerator(rescale=rescale)
    # test_datagen = ImageDataGenerator(rescale=rescale)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory=train_dir, 
        x_col='image_name', 
        y_col='class_id', 
        target_size=target_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=True, 
        seed=random_state
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df, 
        directory=train_dir, 
        x_col='image_name', 
        y_col='class_id', 
        target_size=target_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=False
    )
    return train_gen, val_gen


if __name__ == '__main__':
    print('dataset.py: OK')
