import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import tensorflow_datasets as tfds
from bench_utils import plot_accuracy, plot_loss

# Load and Tweak Model
model_path = "/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/p2_model.h5"

dcv_model = tf.keras.models.load_model(model_path)
dcv_model = tf.keras.models.Sequential(dcv_model.layers[:-1])
dcv_model.add(Dense(10, activation='softmax', name='test_out'))



# Load and Preprocess Data
(data, metadata) = tfds.load('imagenette/320px-v2',split=['train[:90%]', 'train[90%:]', 'validation'],as_supervised=True,with_info=True)

train_ds= data[0]
valid_ds= data[1]
test_ds= data[2]

size = (256, 256)
def preprocess_img(img, label):
  img = img / 255
  img = tf.image.resize(img, size)
  return img, label


train_ds = train_ds.map(preprocess_img).batch(32)
valid_ds = valid_ds.map(preprocess_img).batch(32)
test_ds = test_ds.map(preprocess_img)


# Model Setup
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
dcv_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

history = dcv_model.fit(train_ds, validation_data=valid_ds, epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=[early_stop])

plot_accuracy(history)
plot_loss(history)

results_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Results'
model_path = results_path+'/bench_dcv.h5'
model.save(model_path)
print("Model saved to: ", model_path)