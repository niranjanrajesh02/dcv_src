import matplotlib.pyplot as plt

results_path = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/Results'

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+'/model_loss.png'
    plt.savefig(plot_path)