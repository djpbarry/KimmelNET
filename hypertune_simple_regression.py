import os

import tensorflow as tf
from keras_tuner import BayesianOptimization
from tensorflow import keras
from tensorflow.keras import layers

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 256
epochs = 200
buffer_size = 4
train_path = "Zebrafish_Train_Regression"
test_path = "Zebrafish_Test_Regression"
name = "hypertune_simple_regression"


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image, image_size)
    return image, label


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=image_size + (1,)))
    model.add(layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))
    model.add(layers.experimental.preprocessing.CenterCrop(cropped_image_size[0], cropped_image_size[1]))
    for i in range(4):
        model.add(layers.Conv2D(hp.Int("units_" + str(i), min_value=48, max_value=64, step=16), kernel_size=(3, 3),
                                activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(
        loss="mean_squared_error", optimizer="adam"
    )
    return model


train_list_ds = tf.data.Dataset.list_files(str(train_path + os.sep + "*" + os.sep + "*.png")).shuffle(1000)
train_images_ds = train_list_ds.map(parse_image).batch(batch_size)
val_split = int(0.2 * len(train_images_ds))
val_ds = train_images_ds.take(val_split).prefetch(buffer_size).cache()
train_ds = train_images_ds.skip(val_split).prefetch(buffer_size).cache()
train_ds = train_ds.prefetch(buffer_size=buffer_size).cache()
val_ds = val_ds.prefetch(buffer_size=buffer_size).cache()
test_list_ds = tf.data.Dataset.list_files(str(test_path + os.sep + "*" + os.sep + "*.png")).shuffle(1000)
test_ds = test_list_ds.map(parse_image).batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=buffer_size).cache()

tuner = BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=200
)

tuner.search_space_summary()

tuner.search(train_ds, epochs=epochs, validation_data=val_ds, validation_freq=1)

tuner.results_summary()
