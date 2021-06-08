import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

image_size = (227, 227)
batch_size = 32

#(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

#CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#validation_images, validation_labels = train_images[:5000], train_labels[:5000]
#train_images, train_labels = train_images[5000:], train_labels[5000:]

#train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
#validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Zebrafish",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Zebrafish",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#plt.figure(figsize=(20,20))
#for i, (image, label) in enumerate(train_ds.take(5)):
#    ax = plt.subplot(5,5,i+1)
#    plt.imshow(image)
#    plt.title(CLASS_NAMES[label.numpy()[0]])
#    plt.axis('off')
#plt.show()


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, image_size)
    return image, label


#train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
#test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
#validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

#train_ds_size_1 = tf.data.experimental.cardinality(train_ds_1).numpy()
#validation_ds_size_1 = tf.data.experimental.cardinality(validation_ds_1).numpy()

#print("Training data size:", train_ds_size)
#print("Test data size:", test_ds_size)
#print("Validation data size:", validation_ds_size)

#train_ds = (train_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=batch_size, drop_remainder=True))
#test_ds = (test_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
#validation_ds = (validation_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=batch_size, drop_remainder=True))

root_logdir = os.path.join(os.curdir, "logs\\fit\\")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

#strategy = tf.distribute.MirroredStrategy()

#with strategy.scope():

data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomFlip("vertical"),
    ]
)

inputs = keras.Input(shape=image_size+(3,))
x = data_augmentation(inputs)
x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
x = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
#model.summary()

model.fit(train_ds,
          epochs=1,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])

model.save('trained_alexnet_model')