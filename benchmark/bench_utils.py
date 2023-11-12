import matplotlib.pyplot as plt


results_path = '/home/niranjan.rajesh_asp24/niranjan.rajesh_ug23/DCV/dcv_src/benchmark/Results'

def plot_accuracy(history, model="Model"):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+f'/{model}_accuracy.png'
    plt.savefig(plot_path)
    
def plot_loss(history, model="Model"):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+f'/{model}_loss.png'
    plt.savefig(plot_path)