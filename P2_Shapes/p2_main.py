import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from p2_model import build_model
from p2_utils import plot_loss, plot_accuracy
from p2_dataloader import make_dataset
import argparse

parser = argparse.ArgumentParser(description='DCV Phase 2')

parser.add_argument('--environment', type=str, default='hpc', help='Whether to run on HPC or local machine')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning Rate for Optimizer')
parser.add_argument('--freeze_weights', type=bool, default=True, help='Whether to freeze weights of pretrained layers')
parser.add_argument('--model_name', type=str, default='my_model', help='Name of the saved model file')
parser.add_argument('--new_model', type=bool, default=False, help='Whether to use a new model or load P1 saved model')

args = parser.parse_args()



train_data, valid_data, class_names = make_dataset(args.environment)

results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P2_Shapes/Results'


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model = build_model(len(class_names), args)
history = model.fit(train_data, validation_data=valid_data, epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=[early_stop])
hist_path = results_path+'/history.npy'
np.save(hist_path,history.history)
print("History saved to: ", hist_path)

# summarize history for loss and accuracy
plot_loss(history)
plot_accuracy(history)

model_path = results_path+'/p2_model.h5'
model.save(model_path)
print("Model saved to: ", model_path)