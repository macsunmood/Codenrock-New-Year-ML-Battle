import os
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagesDataset:
    def __init__(self, 
                 data_dir='', 
                 data_csv='', 
                 val_split=0.10, 
                 rescale=1. / 255, 
                 image_size=(300, 300),
                 batch_size=16,
                 random_state=None
                 ):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(data_csv, sep='\t', dtype={'class_id': str})

        self.num_classes = self.data_df['class_id'].nunique()

        # from sklearn.utils import class_weight
        # cl_weights = list(
        #     class_weight.compute_class_weight('balanced', 
        #                                       np.unique(self.data_df['class_id'].values), 
        #                                       self.data_df['class_id'].values)
        # )

        # self.class_weights = dict(enumerate(cl_weights))

        self.rescale = rescale
        self.val_split = val_split
        self.image_size = image_size        
        self.batch_size = batch_size
        self.random_state = random_state        

        self.train_datagen = None

        self.train_gen = None
        self.val_gen = None

        self.test_gen = None

        self.train_df = None
        self.val_df = None

        self.augment_func = self.augment_image

    # Image augmentation object
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5), 

        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, 
                           rotate_limit=35, p=0.5), 

        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.25, 
                                            contrast_limit=0.25), 
                 A.RandomBrightnessContrast(brightness_limit=0.1, 
                                            contrast_limit=0.1)], p=0.2),
        
        A.RandomGamma(p=0.2), 

        A.GaussNoise(p=0.15), 
        
        A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5), p=0.5)
    ])

    def augment_image(self, image):
        '''Applies augmentation to given image'''
        aug_img = np.array(image)
        aug_img = self.augmentation(image=image)['image']
        return aug_img

    def build_generators(self,
                         augment_func=None, 
                         target_size=None, 
                         batch_size=None, 
                         val_split=None, 
                         random_state=None
                         ):
        '''Builds train and val data generators'''
        if not augment_func:
            augment_func = self.augment_func
        if not target_size:
            target_size = self.image_size
        if not batch_size:
            batch_size = self.batch_size
        if not val_split:
            val_split = self.val_split
        if not random_state:
            random_state = self.random_state

        self.train_df, self.val_df = train_test_split(
            self.data_df, 
            test_size=val_split,
            random_state=random_state,
            stratify=self.data_df['class_id'].values
        )

        train_datagen = ImageDataGenerator(preprocessing_function=augment_func, 
                                           rescale=self.rescale)
        val_datagen = ImageDataGenerator(rescale=self.rescale)

        self.train_gen = train_datagen.flow_from_dataframe(
            dataframe=self.train_df, 
            directory=self.data_dir, 
            x_col='image_name', 
            y_col='class_id', 
            target_size=target_size, 
            batch_size=batch_size, 
            class_mode='categorical', 
            shuffle=True, 
            seed=random_state
        )

        self.val_gen = val_datagen.flow_from_dataframe(
            dataframe=self.val_df, 
            directory=self.data_dir, 
            x_col='image_name', 
            y_col='class_id', 
            target_size=target_size, 
            batch_size=batch_size, 
            class_mode='categorical', 
            shuffle=False
        )

    def build_test_generator(self, test_dir):
        '''Builds test data generator from given directory'''
        self.test_dir = test_dir
        self.test_df = pd.DataFrame(os.listdir(test_dir), columns=['image_name'])

        test_datagen = ImageDataGenerator(rescale=self.rescale)
        self.test_gen = test_datagen.flow_from_dataframe(
            dataframe=self.test_df, 
            directory=self.test_dir, 
            x_col='image_name',
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False
        )


if __name__ == '__main__':
    print('dataset.py: OK')
