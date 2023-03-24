import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataloader import DataLoader
from model import Model
from utils import plot_loss

train_data = DataLoader(is_val=False)
val_data = DataLoader(is_val=True)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=2, batch_size=32, verbose=1, callbacks=[early_stop])
np.save('./Results/history1.npy',history.history)

# summarize history for loss
plot_loss(history)
model.save('./Results/my_model.h5')