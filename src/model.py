from tensorflow import keras 
from keras import layers, models

def data_augmentation():
    return models.Sequential([
        layers.RandomFlip("horizontal"),             # horizontal flip
        layers.RandomRotation(0.2),                 # ±20% rotation
        layers.RandomZoom(0.1),                     # ±10% zoom
        layers.RandomTranslation(0.1, 0.1),         # ±10% shift
        layers.RandomContrast(0.1),                 # slight contrast change
        layers.RandomBrightness(0.1)                # slight brightness change
    ])

def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 3))
    inputs = data_augmentation()(inputs) # Create more training data from data augmentation

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(128, activation='relu', name='embedding')(x)
    x = layers.Dropout(0.4)(embedding)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    model.summary()

    return model