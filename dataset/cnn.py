import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

processed_data_path = 'data.pickle'
with open(processed_data_path, 'rb') as file:
    data = pickle.load(file)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

model = tf.keras.Sequential([
    Conv2D(16, (2, 2), input_shape=(400, 400, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(26, activation='softmax')
])

sgd = SGD(learning_rate=1e-2)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


checkpoint = ModelCheckpoint("cnn_model_tf.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32, callbacks=[checkpoint])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")

model.save('./my_model_tf.h5')

print("Model saved successfully.")