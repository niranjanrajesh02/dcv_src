import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import tensorflow_datasets as tfds
from bench_utils import plot_accuracy, plot_loss
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D
from keras.callbacks import CSVLogger


model_path = "/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Models/p2_model.h5"
results_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/v_Results'
data_path = "/storage/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV_data/imagenette2-320/imagenette2-320/train"

# Load and Tweak Model
dcv_model = tf.keras.models.load_model(model_path)
dcv_model = tf.keras.models.Sequential(dcv_model.layers[:-1])
dcv_model.add(Dense(10, activation='softmax', name='test_out'))
vanilla_model = tf.keras.Sequential().from_config(model.get_config())
del dcv_model


# Load and Preprocess Data
data  = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  seed=123,
  shuffle=True,
  subset="both",
  labels="inferred",
  label_mode="int",
  batch_size=None)

train_ds= data[0]
valid_ds= data[1]

size = (256, 256)
def preprocess_img(img, label):
  img = img / 255
  img = tf.image.resize(img, size)
  return img, label


train_ds = train_ds.map(preprocess_img).batch(32)
valid_ds = valid_ds.map(preprocess_img).batch(32)


# Model Setup
csv_logger = CSVLogger(f"{results_path}/dcv_history.csv", append=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
vanilla_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

print("Starting to train model ...")
history = vanilla_model.fit(train_ds, validation_data=valid_ds, epochs=100, batch_size=32, verbose=1, callbacks=[early_stop, csv_logger])
print("Model training complete")


plot_accuracy(history, "Vanilla")
plot_loss(history, "Vanilla")


model_path = results_path+'/bench_vanilla.h5'
vanilla_model.save(model_path)
print("Model saved to: ", model_path)