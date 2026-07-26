import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = (150, 150)
NUM_CLASSES = 6


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Builds a MobileNetV2 transfer-learning model with a custom classification head."""
    base_model = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
    ]


def load_trained_model(model_path):
    """Loads the saved model — used both for prediction and as the base for retraining."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Train a model first.")
    return load_model(model_path)


def retrain_existing_model(model_path, new_data_dir, epochs=5, min_new_images=20, learning_rate=1e-4):
    """
    Retrains the EXISTING saved model (used as a pretrained base) on new data.
    Returns (history, retrained) — retrained=False if the trigger threshold wasn't met.
    """
    from src.preprocessing import count_images_in_dir
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    total_new = count_images_in_dir(new_data_dir)
    if total_new < min_new_images:
        print(f"Only {total_new} new images found (minimum {min_new_images}). Retraining not triggered.")
        return None, False

    print(f"Retraining trigger activated: {total_new} new images found.")
    model = load_trained_model(model_path)

   ALL_CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    new_data_dir, target_size=IMG_SIZE, batch_size=16,
    class_mode='categorical', subset='training',
    classes=ALL_CLASSES
)
val_gen = datagen.flow_from_directory(
    new_data_dir, target_size=IMG_SIZE, batch_size=16,
    class_mode='categorical', subset='validation',
    classes=ALL_CLASSES
)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
    )

    model.save(model_path)
    print(f"Retraining complete. Model updated at {model_path}")
    return history, True