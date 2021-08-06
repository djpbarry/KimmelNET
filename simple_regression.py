import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import numpy as np
import seaborn as sns

image_size = (28, 33)
cropped_image_size = (28, 28)
batch_size = 256
num_classes = 5
epochs = 10000
buffer_size = 4
name = "simple_regression_with_augmentation_and_normalisation"
train_path = "Zebrafish_Train_Regression"
test_path = "Zebrafish_Test_Regression"
#train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression_Subset"
#test_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression_Subset"


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = float(parts[-2])
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


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

plt.figure(num=3, figsize=(20, 17))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy())
        plt.axis("off")
plt.savefig(name + '_sample_images.png')
plt.close(3)

model = keras.Sequential(
    [
        keras.Input(shape=image_size + (1,)),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.experimental.preprocessing.CenterCrop(cropped_image_size[0], cropped_image_size[1]),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1)
    ]
)

model.summary()

model.compile(loss="mean_squared_error", optimizer="adam")

csv_logger = keras.callbacks.CSVLogger(name + '_training.log')

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, validation_freq=1, callbacks=csv_logger)

plt.figure(num=1, figsize=(10, 10))
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig(name + '_training_progress.png')
plt.close(1)

score = model.evaluate(test_ds, verbose=1)
print("Test loss:", score)

labels = np.array([])
predictions = np.array([])
for x, y in test_ds:
    p = model.predict(x, verbose=1)
    for i in range(len(p)):
        predictions = np.concatenate([predictions, p[i]])
    labels = np.concatenate([labels, y.numpy()])

plt.figure(num=2, figsize=(10, 10))
plt.title("Prediction Accuracy")
plt.plot(labels, predictions, 'o')
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.savefig(name + '_prediction_accuracy.png')
