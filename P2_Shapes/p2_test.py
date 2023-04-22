import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def test_loaded_model():
    if env == 'hpc':
        DATA_PATH = '/storage/niranjan.rajesh_ug23/DCV_data/Shapes2D/test'
        MODEL_PATH = '/home/niranjan.rajesh_ug23/DCV/models/best_p2/p2_model.h5'
        RESULTS_PATH = '/home/niranjan.rajesh_ug23/DCV/dcv_src/P2_Shapes/Test_Results'
    elif env == 'local':
        DATA_PATH = 'C:/Niranjan/Ashoka/Research/DCV/Datasets/Shapes2Dummy/dataset/test'
        MODEL_PATH = '"C:/Niranjan/Ashoka/Research/DCV/Models/P2_Shapes1.0/p2_model.h5"'    
        RESULTS_PATH = 'C:/Niranjan/Ashoka/Research/DCV/dcv_src/P2_Shapes/Test_Results/Loaded_Model'
        

    test_data = image_dataset_from_directory(DATA_PATH, labels='inferred', label_mode="categorical", seed=42,
                                                            crop_to_aspect_ratio=True,
                                                            batch_size=32)

    loaded_model =  load_model(MODEL_PATH)

    y_pred = loaded_model.predict(test_data)
    accuracy = accuracy_score(test_data[1], y_pred)

    cm = confusion_matrix(y_test, y_pred, normalize='pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.keys())
    disp.plot()
    _ = disp.ax_.set_title(f'Accuracy: {round(accuracy*100,3)}%')
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'))