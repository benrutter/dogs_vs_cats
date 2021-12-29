from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def train_and_test_iterators():
    train_generator = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True)
    test_generator = ImageDataGenerator(rescale=1.0/255)
    train_iterator = train_generator.flow_from_directory(
        'train',
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
    )
    test_iterator = test_generator.flow_from_directory(
        'test',
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
    )
    return train_iterator, test_iterator


def build_model():
    model = Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(15, 5, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(15, 3, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()],
    )
    return model


if __name__ == '__main__':

    batch_size = 32
    epochs = 25

    early_stop = EarlyStopping(
        monitor='val_auc', mode='min', verbose=1, patience=20)
    train_iterator, test_iterator = train_and_test_iterators()
    model = build_model()

    model.fit(
        train_iterator,
        steps_per_epoch=train_iterator.samples/batch_size,
        epochs=epochs,
        validation_data=test_iterator,
        validation_steps=test_iterator.samples/batch_size,
        callbacks=[early_stop],
    )
