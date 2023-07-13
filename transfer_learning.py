import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 64
epochs = 500
name = 'transfer_learning'

model_dir = "/nemo/stp/lm/working/barryd/hpc/python/zf_reg/outputs/published_model_multi_gpu_custom_augmentation_2023-07-13-12-33-27_0/published_model_multi_gpu_custom_augmentation_trained_model/"
# train_path = "Zebrafish_Train_Regression"
train_parent = "/nemo/stp/lm/working/barryd/hpc/python/keras_image_class/"
# train_path = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression_Augmented"
dataset_path = 'Zebrafish_Train_Princeton_Regression'
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_path = "outputs" + os.sep + name + "_" + date_time + '_' + dataset_path
train_path = train_parent + os.sep + dataset_path

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

# train_list_ds = tf.data.Dataset.from_tensor_slices(filtered_train_files).shuffle(1000)
# train_list_ds = tf.data.Dataset.from_tensor_slices(train_files).shuffle(1000)
dataset = tf.data.Dataset.list_files(train_files)
train_list_ds = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)

print("Number of images in training dataset: ", train_list_ds.cardinality().numpy())

train_images_ds = train_list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
val_split = int(0.2 * len(train_images_ds))
val_ds = train_images_ds.take(val_split).prefetch(tf.data.AUTOTUNE).cache()
train_ds = train_images_ds.skip(val_split).prefetch(tf.data.AUTOTUNE).cache()

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
# checkpointer = keras.callbacks.ModelCheckpoint(filepath=output_path + os.sep + name + '{epoch}', save_best_only=False,
#                                               save_weights_only=True, save_freq=10 * batch_size)

with strategy.scope():
    model = keras.models.load_model(model_dir)
    model.trainable = True

    print('Model has ' + str(len(model.layers)) + ' layers')

    for layer in model.layers[5:-5]:
        layer.trainable = False

    model.summary()

    with open(output_path + os.sep + name + '_model_summary.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write('\n\nTraining Data: ' + train_path)

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, validation_freq=1,
                    callbacks=[csv_logger])

plt.figure(num=1, figsize=(10, 10))
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig(output_path + os.sep + name + '_training_progress.png')
plt.close(1)

model.save(output_path + os.sep + name + '_trained_model')
