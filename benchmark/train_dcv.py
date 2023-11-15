import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import tensorflow_datasets as tfds
from bench_utils import plot_accuracy, plot_loss, augment, preprocess_img
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D
from keras.callbacks import CSVLogger

AUTOTUNE = tf.data.AUTOTUNE


model_path = "/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Models/p2_model.h5"
results_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Results'
data_path = "/storage/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV_data/imagenette2-320/imagenette2-320/train"

# Load and Tweak Model
dcv_model = tf.keras.models.load_model(model_path)
dcv_model = tf.keras.models.Sequential(dcv_model.layers[:-1])
dcv_model.add(Dense(128, activation='relu', name='test_dense'))
dcv_model.add(Dense(10, activation='softmax', name='test_out'))



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

train_ds = data[0]
valid_ds = data[1]



counter = tf.data.experimental.Counter()
train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))
train_ds = (
    train_ds
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_ds
    .batch(32)
    .prefetch(AUTOTUNE)
)


# Model Setup
csv_logger = CSVLogger(f"{results_path}/dcv_history.csv", append=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=10)
dcv_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

print("Starting to train model ...")
history = dcv_model.fit(train_ds, validation_data=valid_ds, epochs=100, batch_size=32, verbose=1, callbacks=[early_stop, csv_logger])
print("Model training complete")


plot_accuracy(history, "DCV")
plot_loss(history, "DCV")

model_path = results_path+'/bench_dcv.h5'
dcv_model.save(model_path)
print("Model saved to: ", model_path)