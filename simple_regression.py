import glob
import os
from datetime import datetime

import matplotlib.colors
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 256
epochs = 500
buffer_size = 4
name = "simple_regression_multi_gpu_added_augmentation"
# train_path = "Zebrafish_Train_Regression"
train_path = "/home/camp/barryd/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = "outputs" + os.sep + name + "_" + date_time

os.makedirs(output_path)

with open(output_path + os.sep + name + '_source.py', 'w') as f:
    f.write(open(__file__).read())


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

train_files = glob.glob(train_path + os.sep + "*" + os.sep + "*.png")
filtered_train_files = [r for r in train_files if
                        "20201127_FishDev_WT_28.5_1-C6" not in r and
                        "20201127_FishDev_WT_28.5_1-H11" not in r and
                        "FishDev_WT_01_1-A3" not in r and
                        "FishDev_WT_01_1-A7" not in r and
                        "FishDev_WT_01_1-D6" not in r and
                        "FishDev_WT_01_1-E3" not in r and
                        "FishDev_WT_01_1-F2" not in r and
                        "FishDev_WT_01_1-G1" not in r and
                        "FishDev_WT_01_1-G5" not in r and
                        "FishDev_WT_01_1-G10" not in r and
                        "FishDev_WT_01_1-H2" not in r and
                        "FishDev_WT_01_1-H8" not in r and
                        "FishDev_WT_02_3-A1" not in r and
                        "FishDev_WT_02_3-A10" not in r and
                        "FishDev_WT_02_3-A4" not in r and
                        "FishDev_WT_02_3-A7" not in r and
                        "FishDev_WT_02_3-C10" not in r and
                        "FishDev_WT_02_3-C11" not in r and
                        "FishDev_WT_02_3-C7" not in r and
                        "FishDev_WT_02_3-D2" not in r and
                        "FishDev_WT_02_3-D6" not in r and
                        "FishDev_WT_02_3-D7" not in r and
                        "FishDev_WT_02_3-D11" not in r and
                        "FishDev_WT_02_3-E1" not in r and
                        "FishDev_WT_02_3-E10" not in r and
                        "FishDev_WT_02_3-E2" not in r and
                        "FishDev_WT_02_3-F12" not in r and
                        "FishDev_WT_02_3-G10" not in r and
                        "FishDev_WT_02_3-G11" not in r and
                        "FishDev_WT_02_3-G12" not in r and
                        "FishDev_WT_02_3-G3" not in r and
                        "FishDev_WT_02_3-G4" not in r and
                        "FishDev_WT_02_3-G8" not in r and
                        "FishDev_WT_02_3-H6" not in r and
                        "FishDev_WT_02_3-H7" not in r]

# train_list_ds = tf.data.Dataset.from_tensor_slices(filtered_train_files).shuffle(1000)
# train_list_ds = tf.data.Dataset.from_tensor_slices(train_files).shuffle(1000)
train_list_ds = tf.data.Dataset.list_files(filtered_train_files).shuffle(1000)

print("Number of images in training dataset: ", train_list_ds.cardinality().numpy())

train_images_ds = train_list_ds.map(parse_image).batch(batch_size)
val_split = int(0.2 * len(train_images_ds))
val_ds = train_images_ds.take(val_split).prefetch(buffer_size).cache()
train_ds = train_images_ds.skip(val_split).prefetch(buffer_size).cache()
train_ds = train_ds.prefetch(buffer_size=buffer_size).cache()
val_ds = val_ds.prefetch(buffer_size=buffer_size).cache()

plt.figure(num=3, figsize=(20, 17))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
        plt.title(labels[i].numpy())
        plt.axis("off")
plt.savefig(output_path + os.sep + name + '_sample_images.png')
plt.close(3)

csv_logger = keras.callbacks.CSVLogger(output_path + os.sep + name + '_training.log')

with strategy.scope():
    model = keras.Sequential(
        [
            keras.Input(shape=image_size + (1,)),
            layers.RandomTranslation(height_factor=0.0, width_factor=0.05),
            layers.RandomFlip(mode="horizontal_and_vertical"),
            layers.RandomContrast(factor=0.1),
            layers.Rescaling(1.0 / 255),
            layers.CenterCrop(cropped_image_size[0], cropped_image_size[1]),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(224, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(112, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(144, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(144, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(1)
        ]
    )

    model.summary()

    with open(output_path + os.sep + name + '_model_summary.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model.compile(loss="mean_squared_error", optimizer="adam")

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, validation_freq=1, callbacks=csv_logger)

plt.figure(num=1, figsize=(10, 10))
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig(output_path + os.sep + name + '_training_progress.png')
plt.close(1)

model.save(output_path + os.sep + name + '_trained_model')
