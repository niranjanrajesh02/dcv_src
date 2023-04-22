import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P2_Shapes/Results'
vanilla_results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P2_Shapes/Vanilla_Results'

def plot_vanilla_loss(history):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = vanilla_results_path +'/model_loss.png'
    plt.savefig(plot_path)
    
def plot_vanilla_accuracy(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = vanilla_results_path +'/model_accuracy.png'
    plt.savefig(plot_path)
    

def plot_loss(history):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+'/model_loss.png'
    plt.savefig(plot_path)
    
def plot_accuracy(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+'/model_accuracy.png'
    plt.savefig(plot_path)
    



def resize_image(image, new_width, new_height):
    # Load the image
    # image = tf.keras.preprocessing.image.load_img(image_path)
    # image = tf.keras.preprocessing.image.img_to_array(image)
    # Resize the image using the Resizing layer
    resized_image = tf.keras.layers.Resizing(new_height, new_width)(image)

    return resized_image