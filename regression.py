import tensorflow
from tensorflow import keras

from sklearn import preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

x_train_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
x_test_scaled = scaler.transform(x_test)

model = keras.Sequential()
model.add(keras.layers.Dense(64, kernel_initializer='normal', activation='relu', input_shape=(13,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(
    loss='mse',
    optimizer=keras.optimizers.RMSprop(),
    metrics=['mean_absolute_error']
)

history = model.fit(
    x_train_scaled, y_train,
    batch_size=128,
    epochs=500,
    verbose=1,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
)

score = model.evaluate(x_test_scaled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
