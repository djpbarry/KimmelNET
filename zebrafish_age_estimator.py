import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import definitions

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 256
name = definitions.test_source_folder + "_" + definitions.name
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# output_path = "outputs" + os.sep + name + "_" + date_time
parent_model_path = '/nemo/stp/lm/working/barryd/hpc/python/zf_reg/outputs/'
data_path = '/nemo/stp/lm/working/barryd/hpc/python//keras_image_class/'
model_list = glob.glob(parent_model_path + os.sep + sys.argv[2])
model_path = model_list[int(sys.argv[1])]
datasets = ('Zebrafish_Test_Regression', 'Zebrafish_25C', 'Zebrafish_Test_Princeton_Regression', '20232803 ZF 15 mins 25')


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label, filename


print(date_time + ': Let\'s go!')

model_name = os.path.basename(model_path)
print('Testing model ' + model_name)
for data in datasets:
    print('Testing dataset ' + data)
    output_path = parent_model_path + os.sep + model_name + os.sep + 'test_outputs' + os.sep + data
    os.makedirs(output_path)

    test_path = data_path + os.sep + data
    # test_path = "./test_data"

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
    # test_list_ds = tf.data.Dataset.from_tensor_slices(filtered_test_files).shuffle(1000)

    print("Number of images in test dataset: ", test_list_ds.cardinality().numpy())

    test_ds = test_list_ds.map(parse_image).batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE).cache()

    model_dir = glob.glob(parent_model_path + os.sep + model_name + os.sep + '*model')

    model = keras.models.load_model(model_dir[0])

    # latest = tf.train.latest_checkpoint('/nemo/stp/lm/working/barryd/hpc/python/zf_reg/outputs/simple_regression_multi_gpu_added_augmentation_2023-04-21-15-45-29')

    # model.load_weights(latest)

    # model = keras.models.load_model('./simple_regression_trained_model')

    model.summary()

    # score = model.evaluate(test_ds, verbose=1)
    # print("Test loss:", score)

    labels = np.array([])
    predictions = np.array([])
    files = np.array([])
    for x, y, z in test_ds:
        p = model.predict(x, verbose=1)
        for i in range(len(p)):
            predictions = np.concatenate([predictions, p[i]])
        labels = np.concatenate([labels, y.numpy()])
        files = np.concatenate([files, [f.decode() for f in z.numpy()]])

    print(date_time + ': Done')

    linear_model = np.polyfit(labels, predictions, 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s = np.arange(4, 53)

    plt.figure(num=2, figsize=(10, 10))
    plt.title("Prediction Accuracy")
    plt.plot(labels, predictions, 'o', markersize=3)
    plt.plot(x_s, linear_model_fn(x_s), color="red")
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig(output_path + os.sep + name + '_prediction_accuracy.png')

    dictionary = {'Label': labels, 'Prediction': predictions, 'File': files}
    dataFrame = pd.DataFrame(dictionary)
    dataFrame.to_csv(output_path + os.sep + name + '_predictions.csv')

    errs = labels - predictions

    plt.figure(num=4, figsize=(10, 10))
    plt.title("Prediction Errors")
    plt.hist(errs, bins=100)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.savefig(output_path + os.sep + name + '_prediction_errors.png')
