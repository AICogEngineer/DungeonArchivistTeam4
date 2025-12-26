from tensorflow import keras 
from keras import layers, models

def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    embedding = layers.Dense(64, activation='relu', name='embedding')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(embedding)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    model.summary()

    return model