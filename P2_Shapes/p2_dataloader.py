import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



shape_count = {}

def rename_files():
    for file in os.listdir(DATA_PATH):
        shape = file.split('_')[0]
        
        if shape not in shape_count.keys():
            shape_count[shape] = 1
        else:
            shape_count[shape] += 1
        
        new_name = shape + '_' + str(shape_count[shape]).zfill(4) + '.png'
        src = os.path.join(DATA_PATH, file)
        dest = os.path.join(DATA_PATH, new_name)
        os.rename(src, dest)

def make_subdirs():
    # create a folder for each class in the dataset and move the images to the corresponding folder
    for file in os.listdir(DATA_PATH):
        shape = file.split('_')[0]
        src = os.path.join(DATA_PATH, file)
        dest = os.path.join(DATA_PATH, shape)

        if not os.path.exists(dest):
            os.makedirs(dest)
        os.rename(src, os.path.join(dest, file))

def make_dataset(env):
    if env == 'hpc':
        DATA_PATH = '/storage/niranjan.rajesh_ug23/DCV_data/Shapes2D/output'
    elif env == 'local':
        DATA_PATH = 'C:/Niranjan/Ashoka/Research/DCV/Datasets/Shapes2Dummy/dataset/output'
    
    # rename_files()
    # make_subdirs()
        
    train_data = image_dataset_from_directory(DATA_PATH, labels='inferred', label_mode="categorical", seed=42,
                                                         validation_split=0.2, subset="training", crop_to_aspect_ratio=True,
                                                         batch_size=32)
    valid_data = image_dataset_from_directory(DATA_PATH, labels='inferred', label_mode="categorical", seed=42,
                                                         validation_split=0.2, subset="validation", crop_to_aspect_ratio=True,
                                                         batch_size=32)
    
    
    class_names = train_data.class_names
    print("Class Names: ", class_names)
    
    return train_data, valid_data, class_names


    
    
    
if __name__ == "__main__":
    
    train_data, valid_data, class_names = make_dataset('local')
    image_batch, label_batch = next(iter(train_data))
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = label_batch[i]
        # plt.title(class_names[label])
        print(label)
        plt.axis("off")
        plt.savefig("C:/Niranjan/Ashoka/Research/DCV/dcv_src/P2_Shapes/data_vis.jpg")

    print(class_names)
