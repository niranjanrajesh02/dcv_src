import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataloader import DataLoader
from model import Model
from utils import plot_loss

train_data = DataLoader(is_val=False)
val_data = DataLoader(is_val=True)

results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/Results'

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=2, batch_size=32, verbose=1, callbacks=[early_stop])
hist_path = results_path+'/history.npy'
np.save(hist_path,history.history)

# summarize history for loss
plot_loss(history)
model_path = results_path+'/my_model.h5'
model.save(model_path)