import matplotlib.pyplot as plt
import tensorflow as tf


dcv_results_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Results'
vanilla_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/v_Results'

def plot_accuracy(history, model="Model"):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    if model == "DCV":
        plot_path = dcv_results_path+f'/{model}_accuracy.png'
    elif model == "Vanilla":
        plot_path = vanilla_path+f'/{model}_accuracy.png'
    plt.savefig(plot_path)
    
def plot_loss(history, model="Model"):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    if model == "DCV":                  
        plot_path = dcv_results_path+f'/{model}_loss.png'
    elif model == "Vanilla":
        plot_path = vanilla_path+f'/{model}_loss.png'
    plt.savefig(plot_path) 
    
def augment(image_label, seed):
  image, label = image_label
  new_seed = tf.random.split(seed, num=1)[0, :]
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.2, seed=new_seed)
  # Random rotation
  image = tf.image.stateless_random_flip_left_right(image, seed)
  image = tf.image.stateless_random_flip_up_down(image, seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

size = (256, 256)
def preprocess_img(img, label):
  img = img / 255
  img = tf.image.resize(img, size)
  return img, label