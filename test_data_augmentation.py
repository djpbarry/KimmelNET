import glob
import os
from datetime import datetime

import numpy as np
import skimage as sk
import tensorflow as tf
from keras import layers
from tensorflow import keras

import definitions

hist_eq_thresh = np.random.default_rng().choice(np.array(range(1, 10)) / 10.0)
sat_thresh = np.random.default_rng().choice(np.array(range(1, 10)) / 10.0)
noise_thresh = np.random.default_rng().choice(np.array(range(1, 10)) / 10.0)
noise_sd = np.random.default_rng().choice(np.array(range(1, 11)) / 500.0)
percent_1 = np.random.default_rng().choice(np.array(range(1, 11)))
percent_2 = np.random.default_rng().choice(np.array(range(1, 11)))

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 32
buffer_size = 1
name = "simple_regression_" + definitions.name

# train_path = "Zebrafish_Train_Regression"
train_path = "/nemo/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression"
aug_path = "/nemo/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression_Augmented"
# train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression/"
# train_path = "C:/Users/davej/Dropbox (The Francis Crick)/ZF_Test"
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
output_path = aug_path + '_' + date_time

os.makedirs(output_path)

for i in range(190):
    if not os.path.exists(output_path + os.sep + str(4.5 + i * 0.25)):
        os.makedirs(output_path + os.sep + str(4.5 + i * 0.25))

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


def random_augment_img(x, p=0.25):
    x = x.numpy() / 255.0
    if np.random.default_rng().uniform() > hist_eq_thresh:
        x = sk.exposure.equalize_adapthist(x)
    if np.random.default_rng().uniform() > sat_thresh:
        v_min, v_max = np.percentile(x,
                                     (np.random.default_rng().uniform() * percent_1,
                                      100.0 - np.random.default_rng().uniform() * percent_2))
        x = sk.exposure.rescale_intensity(x, in_range=(v_min, v_max))
    if np.random.default_rng().uniform() > noise_thresh:
        x = sk.util.random_noise(x, var=noise_sd * np.random.default_rng().uniform())
    x = tf.convert_to_tensor(255.0 * x)
    return x


def random_augment(factor=0.5):
    return layers.Lambda(lambda x: random_augment_img(x, factor))


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

train_images_ds = train_list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
train_ds = train_images_ds.prefetch(tf.data.AUTOTUNE).cache()
# train_ds = train_ds.prefetch(buffer_size=buffer_size).cache()
# val_ds = val_ds.prefetch(buffer_size=buffer_size).cache()

fill = 'reflect'
inter = 'bilinear'

with strategy.scope():
    model = keras.Sequential(
        [
            #            layers.RandomFlip(mode="horizontal_and_vertical"),
            #           layers.RandomTranslation(height_factor=0.0, width_factor=0.1, fill_mode=fill,
            #                                    interpolation=inter),
            #           layers.RandomZoom(height_factor=(-0.3, 0.0), fill_mode=fill, interpolation=inter),
            layers.RandomBrightness(factor=(-0.1, 0.3)),
            random_augment(factor=0.5),
            #           layers.Rescaling(1.0 / 255)
        ]
    )

# plt.figure(figsize=(20, 17))

# result = model(train_ds.take(1))

count = 0

for images, labels in train_ds:
    for i in range(len(images)):
        augImage = model(images[i], training=True).numpy()
        sk.io.imsave(output_path + os.sep + str(labels[i].numpy()) + os.sep + str(count) + '_augmented.png',
                     augImage[:, :, 0].astype('ubyte'))
        count = count + 1

model.summary()

with open(output_path + os.sep + name + '_model_summary.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.write('hist_eq_thresh: ' + str(hist_eq_thresh))
    fh.write('sat_thresh: ' + str(sat_thresh))
    fh.write('noise_thresh: ' + str(noise_thresh))
    fh.write('noise_sd: ' + str(noise_sd))
    fh.write('percent_1: ' + str(percent_1))
    fh.write('percent_2: ' + str(percent_2))

print("Done.")
