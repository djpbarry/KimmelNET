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
epochs = 500
buffer_size = 4
name = "simple_classifier_with_augmentation_and_normalisation_and_batchnormalisation"
CLASS_NAMES = ['Early', 'Early-Mid', 'Mid', 'Late-Mid', 'Late']

# the data, split between train and test sets
train_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Train",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

validation_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Train",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Test",
    # "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test",
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    labels="inferred",
    label_mode="categorical"
)

train_ds = train_ds.prefetch(buffer_size=buffer_size).cache()
validation_ds = validation_ds.prefetch(buffer_size=buffer_size).cache()

plt.figure(figsize=(20, 17))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        for j in range(len(CLASS_NAMES)):
            if labels.numpy()[i][j] > 0:
                plt.title(CLASS_NAMES[j])
        plt.axis("off")
plt.savefig(name + '_sample_images.png')

model = keras.Sequential(
    [
        keras.Input(shape=image_size + (1,)),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.experimental.preprocessing.CenterCrop(cropped_image_size[0], cropped_image_size[1]),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

csv_logger = keras.callbacks.CSVLogger(name + '_training.log')

history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, validation_freq=1, callbacks=csv_logger)

plt.figure(num=1, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig(name + '_classification_progress.png')
plt.close(1)

score = model.evaluate(test_ds, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

labels = np.array([])
predictions = np.array([])
for x, y in test_ds:
    predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=1), axis=1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=1)])

confusion = sk_metrics.confusion_matrix(labels, predictions)
confusion_normalized = np.transpose(np.transpose(confusion.astype("float")) / confusion.sum(axis=1))
axis_labels = list(CLASS_NAMES)
plt.figure(num=2, figsize=(10, 10))
sns.heatmap(
    confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
    cmap='Blues', annot=True, fmt='.2f', square=True)
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(name + '_confusion_matrix.png')
