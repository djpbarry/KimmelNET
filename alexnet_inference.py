from tensorflow.keras.models import load_model

model = load_model('trained_alexnet_model')
model.summary()