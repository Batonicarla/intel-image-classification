import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def get_train_val_generators(train_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """Returns (train_generator, val_generator) with augmentation applied to training data."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training', shuffle=True, seed=42
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation', shuffle=False, seed=42
    )
    return train_gen, val_gen


def get_test_generator(test_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """Returns a generator for the untouched test set (no augmentation, only rescaling)."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    return test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )


def preprocess_single_image(img_path, img_size=IMG_SIZE):
    """Loads and preprocesses a single image file for prediction."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def count_images_in_dir(data_dir):
    """Counts total images across class subfolders — used by the retraining trigger."""
    total = 0
    if not os.path.isdir(data_dir):
        return 0
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            total += len(os.listdir(cls_path))
    return total