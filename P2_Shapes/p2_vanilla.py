import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from p2_model import build_model
from p2_utils import plot_loss, plot_accuracy
from p2_dataloader import make_dataset


def train_vanilla(args):
    train_data, valid_data, class_names = make_dataset(args.environment)

    results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P2_Shapes/Vanilla_Results'

    model = build_model(len(class_names), args)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(train_data, validation_data=valid_data, epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=[early_stop])
    hist_path = results_path+'/history.npy'
    np.save(hist_path,history.history)
    print("History saved to: ", hist_path)

    # summarize history for loss and accuracy
    plot_loss(history)
    plot_accuracy(history)

    model_path = results_path+'/p2_van_model.h5'
    model.save(model_path)
    print("Vanilla Model saved to: ", model_path)