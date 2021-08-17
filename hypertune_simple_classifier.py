from keras_tuner import Hyperband
from tensorflow import keras
from tensorflow.keras import layers

image_size = (112, 134)
cropped_image_size = (112, 112)
batch_size = 256
num_classes = 5
epochs = 500
buffer_size = 4
name = "hypertune_simple_classifier"


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=image_size + (1,)))
    model.add(layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1))
    model.add(layers.experimental.preprocessing.RandomRotation(factor=0.1))
    model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255))
    model.add(layers.experimental.preprocessing.CenterCrop(cropped_image_size[0], cropped_image_size[1]))
    for i in range(hp.Int("num_layers", 2, 5)):
        model.add(layers.Conv2D(hp.Int("units_" + str(i), min_value=16, max_value=256, step=16), kernel_size=(3, 3),
                                activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(hp.Choice("rate", [0.5, 0.6, 0.7])))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
    )
    return model


# the data, split between train and test sets
train_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Train",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

validation_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Train",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Test",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test",
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

train_ds = train_ds.prefetch(buffer_size=buffer_size).cache()
validation_ds = validation_ds.prefetch(buffer_size=buffer_size).cache()

tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=200,
    factor=3,
    overwrite=True,
    directory="outputs",
    project_name=name,
)

tuner.search_space_summary()

tuner.search(train_ds, epochs=epochs, validation_data=validation_ds, validation_freq=1)

tuner.results_summary()
