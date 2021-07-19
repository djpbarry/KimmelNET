import datetime
import os

import matplotlib.pyplot as plt
import numpy
import pandas
from tensorflow import keras

image_size = (190, 227)
batch_size = 128
# num_classes = 5
epochs = 200
# buffer_size = 2
train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
test_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression"


# train_path = "Zebrafish_Train_Regression"
# test_path = "Zebrafish_Test_Regression"


def load_images(inputDir, size):
    # images = []
    paths = []
    start = len(inputDir) + 1
    for folder_name in os.listdir(inputDir):
        current_dir = os.path.join(inputDir, folder_name)
        print(str(datetime.datetime.now()) + ": Processing " + current_dir + " - Number of files: " + str(len(os.listdir(current_dir))))
        for fname in os.listdir(current_dir):
            fpath = os.path.join(current_dir, fname)
            #            image = skio.imread(fpath)
            #            image = skt.resize(image, size)
            #            images.append(image)
            hpfs = float(fpath[start:fpath.index("\\", start)])
            paths.append({'File': fpath, 'hpfs': hpfs})
    # return numpy.array(images), paths
    return paths


# train_images, train_paths = load_images(train_path, image_size)
train_paths = load_images(train_path, image_size)
start = len(train_path) + 1
# hpfs = numpy.array([[float(f[start:index]) for index in [f.index("\\", start)]] for f in train_paths])

inputDataFrame = pandas.DataFrame(train_paths)

plt.hist(inputDataFrame.hpfs, bins=190)
plt.show()

# print(inputDataFrame[1:20])

train_datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

trainImageGen = train_datagen.flow_from_dataframe(inputDataFrame, x_col='File', y_col='hpfs', target_size=image_size,
                                                  class_mode='raw', color_mode='grayscale', batch_size=batch_size,
                                                  subset="training", directory=None)

valImageGen = train_datagen.flow_from_dataframe(inputDataFrame, x_col='File', y_col='hpfs', target_size=image_size,
                                                class_mode='raw', color_mode='grayscale', batch_size=batch_size,
                                                subset="validation", directory=None)
# train_images, val_images, train_hpfs, val_hpfs = sklms.train_test_split(train_images, hpfs, test_size=0.2)

# auto = tf.data.AUTOTUNE
# train_ds = train_ds.prefetch(buffer_size=auto).cache()
# test_ds = test_ds.prefetch(buffer_size=auto).cache()

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

history = model.fit(trainImageGen,
                    epochs=epochs,
                    validation_freq=1,
                    validation_data=valImageGen)

#print(history.history.keys())

model.save('trained_alexnet_regression_model')

plt.figure(figsize=(20, 10))
#plt.subplot(1, 2, 1)
#plt.suptitle('Optimizer : Adam', fontsize=10)
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

#plt.subplot(1, 2, 2)
#plt.ylabel('Mean Absolute Error', fontsize=16)
#plt.plot(history.history['mean_absolute_error'], label='Training Error')
#plt.plot(history.history['val_mean_absolute_error'], label='Validation Error')
#plt.legend(loc='lower right')
plt.savefig('regression_training_progress.png')

#test_images, test_paths = load_images(test_path, image_size)
#start = len(test_path) + 1
#test_hpfs = numpy.array([[float(f[start:index]) for index in [f.index("/", start)]] for f in test_paths])

#model.evaluate(test_images, test_hpfs)
