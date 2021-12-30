from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def train_and_test_iterators(batch_size):
    train_generator = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.1)
    test_generator = ImageDataGenerator(rescale=1.0/255)
    train_iterator = train_generator.flow_from_directory(
        'data/train',
        target_size=(120, 120),
        batch_size=batch_size,
        class_mode='binary',
    )
    test_iterator = test_generator.flow_from_directory(
        'data/test',
        target_size=(120, 120),
        batch_size=batch_size,
        class_mode='binary',
    )
    return train_iterator, test_iterator


def build_model():
    model = Sequential()
    model.add(keras.Input(shape=(120, 120, 3)))
    model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':

    batch_size = 32
    epochs = 25

    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
    )

    train_iterator, test_iterator = train_and_test_iterators(batch_size)
    model = build_model()

    model.fit(
        train_iterator,
        steps_per_epoch=int(train_iterator.samples/batch_size),
        epochs=epochs,
        validation_data=test_iterator,
        validation_steps=int(test_iterator.samples/batch_size),
        callbacks=[early_stop],
    )

    model.save_model('cnn')
