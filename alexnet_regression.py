import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy
import sklearn
import skimage.io as skio
import skimage.transform as skt
import os
import time

image_size = (190, 227)
batch_size = 32
# num_classes = 5
epochs = 50
# buffer_size = 2
train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
test_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression"

def load_images(inputDir, size):
    images = []
    paths = []
    for folder_name in os.listdir(inputDir):
        print("Processing " + os.path.join(inputDir, folder_name))
        for fname in os.listdir(os.path.join(inputDir, folder_name)):
            fpath = os.path.join(inputDir, folder_name, fname)
            image = skio.imread(fpath)
            image = skt.resize(image, size)
            images.append(image)
            paths.append(fpath)
    return numpy.array(images), paths


images, paths = load_images(train_path, image_size)

start = len(train_path) + 1
hpfs = numpy.array([[float(f[start:index]) for index in [f.index("\\", start)]] for f in paths])

train_images, val_images, train_hpfs, val_hpfs = sklearn.model_selection.train_test_split(images, hpfs, test_size=0.2)

test_images, test_paths = load_images(test_path, image_size)
start = len(test_path) + 1
test_hpfs = numpy.array([[float(f[start:index]) for index in [f.index("\\", start)]] for f in test_paths])

#auto = tf.data.AUTOTUNE
#train_ds = train_ds.prefetch(buffer_size=auto).cache()
#test_ds = test_ds.prefetch(buffer_size=auto).cache()

# plt.figure(figsize=(20, 20))
# for images, labels in train_ds.take(1):
#    for i in range(25):
#    ax = plt.subplot(5, 5, i + 1)
#   plt.imshow(images[i].numpy().astype("uint8"))
#    for j in range(len(CLASS_NAMES)):
#         if labels.numpy()[i][j] > 0:
#              plt.title(CLASS_NAMES[j])
#       plt.axis("off")
# plt.savefig('sample_images.png')


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
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(
    loss='mse',
    optimizer=keras.optimizers.RMSprop(),
    metrics=['mean_absolute_error']
)

history = model.fit(train_images, train_hpfs,
                    epochs=epochs,
                    validation_freq=1,
                    validation_data=(val_images, val_hpfs))

model.evaluate(test_images, test_hpfs)
