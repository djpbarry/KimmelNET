import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 256
buffer_size = 4
name = "zf_regression_test"
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = "outputs" + os.sep + name + "_" + date_time

os.makedirs(output_path)

def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label, filename


test_path = "/home/camp/barryd/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression"
#test_path = "./test_data"

test_list_ds = tf.data.Dataset.list_files(str(test_path + os.sep + "*" + os.sep + "*.png")).shuffle(1000)
test_ds = test_list_ds.map(parse_image).batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=buffer_size).cache()

model = keras.models.load_model('/home/camp/barryd/working/barryd/hpc/python/keras_image_class/outputs'
                                '/simple_regression_2021-08-23-14-39-08/simple_regression_trained_model')

#model = keras.models.load_model('./simple_regression_trained_model')

model.summary()

#score = model.evaluate(test_ds, verbose=1)
#print("Test loss:", score)

labels = np.array([])
predictions = np.array([])
files = np.array([])
for x, y, z in test_ds:
    p = model.predict(x, verbose=1)
    for i in range(len(p)):
        predictions = np.concatenate([predictions, p[i]])
    labels = np.concatenate([labels, y.numpy()])
    files = np.concatenate([files, z.numpy()])

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

np.savetxt(output_path + os.sep + name + '_predictions.csv', np.transpose(np.concatenate([[labels.astype('S')],
                                                                                          [predictions.astype('S')],
                                                                                          [files.astype('S')]])),
           delimiter=',', fmt='%s', header='Label,Prediction,File', encoding='UTF-8')

errs = labels - predictions

plt.figure(num=4, figsize=(10, 10))
plt.title("Prediction Errors")
plt.hist(errs, bins=100)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.savefig(output_path + os.sep + name + '_prediction_errors.png')
