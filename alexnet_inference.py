import tensorflow as tf
from tensorflow.keras.models import load_model

image_size = (227, 227)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Zebrafish_Test",
    image_size=image_size,
    batch_size=batch_size,
)

model = load_model('trained_alexnet_model')
model.summary()

model.evaluate(test_ds)