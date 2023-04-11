import os
import glob
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

import definitions

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 16
buffer_size = 4
name = definitions.test_source_folder + "_" + definitions.name
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = "outputs" + os.sep + name + "_" + date_time

os.makedirs(output_path)


def prep_input(path):
    image = tf.image.decode_png(tf.io.read_file(path))
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, image_size)
    return image

def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label, filename


def norm_flat_image(img):
    grads_norm = img[:, :, 0]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm)) / (tf.reduce_max(grads_norm) - tf.reduce_min(grads_norm))
    return grads_norm


def plot_maps(img1, img2, vmin=0.3, vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(45, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(img1 * mix_val + img2 / mix_val, cmap="gray")
    plt.axis("off")
    plt.show()


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy

    return tf.nn.relu(x), grad


model = keras.models.load_model('/nemo/stp/lm/working/barryd/hpc/python/zf_reg/outputs'
                                '/simple_regression_multi_gpu_added_augmentation_2022-07-04-13-07-05'
                                '/simple_regression_multi_gpu_added_augmentation_trained_model')

layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
for layer in layer_dict:
    if layer.activation == tf.keras.activations.relu:
        layer.activation = guidedRelu
        print("changed")

# model = keras.models.load_model('./simple_regression_trained_model')

model.summary()

test_path = "/nemo/stp/lm/working/barryd/hpc/python/keras_image_class/" + definitions.test_source_folder

test_files = glob.glob(test_path + os.sep + "*" + os.sep + "*.png")

filtered_test_files = [r for r in test_files if
                       "FishDev_WT_25C_1-B2" not in r and
                       "FishDev_WT_25C_1-B4" not in r and
                       "FishDev_WT_25C_1-B7" not in r and
                       "FishDev_WT_25C_1-B9" not in r and
                       "FishDev_WT_25C_1-D8" not in r and
                       "FishDev_WT_25C_1-E9" not in r and
                       "FishDev_WT_25C_1-F8" not in r and
                       "FishDev_WT_25C_1-G7" not in r and
                       "FishDev_WT_25C_1-G11" not in r and
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

test_list_ds = tf.data.Dataset.list_files(filtered_test_files).shuffle(1000)

print("Number of images in training dataset: ", test_list_ds.cardinality().numpy())

test_ds = test_list_ds.map(parse_image).batch(batch_size)
test_ds = test_ds.cache().prefetch(buffer_size=buffer_size)


#input_img = prep_input('Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression/45'
#                       '/20201127_FishDev_WT_28.5_1-C10-163.png')

for x, y, z in test_ds:
    with tf.GradientTape() as tape:
        tape.watch(x)
        result = model(x)
    grads = tape.gradient(result, x)
    filenames = [f.decode() for f in z.numpy()]
    for g in range(len(grads)):
        np.savetxt(output_path + os.sep + os.path.split(filenames[g])[1] + '_saliency_map.txt', grads[g, :, :, 0])

#plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0]))

#plt.imshow(norm_flat_image(grads[0]))
#plt.savefig(output_path + os.sep + name + '_saliency_map.png', cmap="gray")

#tf.keras.utils.save_img(output_path + os.sep + name + '_saliency_map.tif', grads[0])

