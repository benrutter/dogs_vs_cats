from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input


def train_and_test_iterators(batch_size):
    train_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    test_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    train_iterator = train_generator.flow_from_directory(
        'data/train',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
    )
    test_iterator = test_generator.flow_from_directory(
        'data/test',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
    )
    return train_iterator, test_iterator


pretrained_model = ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
)

for layer in pretrained_model.layers:
    layer.trainable = False

def build_model():
    x = keras.layers.GlobalAveragePooling2D()(
        pretrained_model.get_layer('conv5_block3_out').output
    )
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(pretrained_model.input, x)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


if __name__ == '__main__':

    batch_size = 64
    epochs = 5

    train_iterator, test_iterator = train_and_test_iterators(batch_size)
    model = build_model()

    model.fit(
        train_iterator,
        steps_per_epoch=int(train_iterator.samples/batch_size),
        epochs=epochs,
        validation_data=test_iterator,
        validation_steps=int(test_iterator.samples/batch_size),
    )

    model.save('cnn')
