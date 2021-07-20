import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

image_size = (190, 227)
batch_size = 32
# num_classes = 5
epochs = 200
buffer_size = 4
# train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
# test_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression"


train_path = "Zebrafish_Train_Regression"
test_path = "Zebrafish_Test_Regression"


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


list_ds = tf.data.Dataset.list_files(str(train_path + os.sep + "*" + os.sep + "*.png")).shuffle(1000)
images_ds = list_ds.map(parse_image).batch(batch_size)
val_split = int(0.2 * len(images_ds))
val_ds = images_ds.take(val_split).prefetch(buffer_size).cache()
train_ds = images_ds.skip(val_split).prefetch(buffer_size).cache()

# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, fill_mode="reflect",
                                                                  interpolation="bilinear"),
        keras.layers.experimental.preprocessing.RandomRotation(1.0, fill_mode="reflect", interpolation="bilinear")
    ]
)

inputs = keras.Input(shape=image_size + (1,))
x = data_augmentation(inputs)
# x = keras.layers.experimental.preprocessing.Resizing(190, 227)(x)
x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
x = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1, activation="linear")(x)
model = keras.Model(inputs, outputs)

model.compile(
    loss='mean_absolute_error',
    optimizer=keras.optimizers.Adam()
)

history = model.fit(train_ds,
                    epochs=epochs,
                    validation_freq=1,
                    validation_data=val_ds)

# print(history.history.keys())

model.save('trained_alexnet_regression_model')

plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.suptitle('Optimizer : Adam', fontsize=10)
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

# plt.subplot(1, 2, 2)
# plt.ylabel('Mean Absolute Error', fontsize=16)
# plt.plot(history.history['mean_absolute_error'], label='Training Error')
# plt.plot(history.history['val_mean_absolute_error'], label='Validation Error')
# plt.legend(loc='lower right')
plt.savefig('regression_training_progress.png')

# test_images, test_paths = load_images(test_path, image_size)
# start = len(test_path) + 1
# test_hpfs = numpy.array([[float(f[start:index]) for index in [f.index("/", start)]] for f in test_paths])

# model.evaluate(test_images, test_hpfs)
