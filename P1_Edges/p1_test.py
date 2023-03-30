import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
new_model = load_model('C:/Niranjan/Ashoka/Research/DCV/dcv_src/P1_Edges/Results/my_model.h5')

test_data_path = "C:/Niranjan/Ashoka/Research/DCV/Datasets/BIPED/edges/imgs/test/rgbr/"
results_path = "C:/Niranjan/Ashoka/Research/DCV/dcv_src/P1_Edges/Results/Test_Output/"

test_imgs = os.listdir(test_data_path)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for img in test_imgs:
    img_x = cv.imread(test_data_path+img)
    pred = new_model.predict(np.expand_dims(img_x, axis=0),batch_size=1)
    
    axs[0].imshow(img_x)
    axs[0].axis('off')
    axs[0].set_title('Input Image')
    axs[1].imshow(pred[0], cmap='Greys')
    axs[1].axis('off')
    axs[1].set_title('Predicted Edge Map')
    
    plt.savefig(results_path+img)
    print("Image saved: ", img)
    
print("Inference Complete! Images saved in: ", results_path)

    


# img_x = cv.imread("C:/Niranjan/Ashoka/Research/DCV/Datasets/BIPED/edges/imgs/test/rgbr/RGB_017.jpg")

# pred = new_model.predict(np.expand_dims(img_x, axis=0),batch_size=1)

# fig = plt.imshow(pred[0], cmap='Greys')

# plt.savefig("C:/Niranjan/Ashoka/Research/DCV/dcv_src/P1_Edges/Results/pred.png")
# plt.close(fig)