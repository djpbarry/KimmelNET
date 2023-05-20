import glob
import os
from datetime import datetime

import skimage as sk
import tensorflow as tf
from keras import layers
from tensorflow import keras

import definitions

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 32
buffer_size = 1
name = "simple_regression_" + definitions.name

# train_path = "Zebrafish_Train_Regression"
# train_path = "/nemo/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression/"
#train_path = "C:/Users/davej/Dropbox (The Francis Crick)/ZF_Test"
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

train_files = glob.glob(train_path + os.sep + "4.5" + os.sep + "*.png")
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

fill = 'reflect'
inter = 'bilinear'

model = keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical"),
        layers.RandomTranslation(height_factor=0.0, width_factor=0.1, fill_mode=fill,
                                 interpolation=inter),
        #layers.RandomRotation(factor=0.1, fill_mode=fill, interpolation=inter),
        layers.RandomZoom(height_factor=(-0.3, 0.1), fill_mode=fill, interpolation=inter),
        layers.GaussianNoise(stddev=10.0),
        layers.RandomContrast(factor=0.75),
        layers.RandomBrightness(factor=0.5),
        layers.Rescaling(1.0 / 255)
    ]
)

# plt.figure(figsize=(20, 17))

# result = model(train_ds.take(1))

for images, labels in train_ds.take(1):
    for i in range(25):
        augImage = model(images[i], training=True).numpy()
        sk.io.imsave(output_path + os.sep + str(labels[i].numpy()) + '_' + str(i) + '_augmented.tiff',
                     augImage[:, :, 0])
        sk.io.imsave(output_path + os.sep + str(labels[i].numpy()) + '_' + str(i) + '_original.tiff',
                     images[i].numpy())

model.summary()

with open(output_path + os.sep + name + '_model_summary.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("Done.")
