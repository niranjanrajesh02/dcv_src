import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input



def build_model(num_classes, args):
      
  env = args.environment
  new_model = args.new_model
  
  if env == 'hpc':
        P1_MODEL_PATH = "/home/niranjan.rajesh_ug23/DCV/models/best_p1/my_model.h5"
  elif env == 'local':
        P1_MODEL_PATH = "C:/Niranjan/Ashoka/Research/DCV/Models/P1_Edges3.0/my_model.h5"
        
        

      
#   model = tf.keras.models.load_model(P1_MODEL_PATH)

#   # removing the deconv and upsampling layers
#   model = tf.keras.models.Sequential(model.layers[:-5])
  
#   if not new_model:
#       for layer in model.layers:
#             layer.trainable = False

#   model.add(Conv2D(32, (3,3), padding='same', name='p2_conv_1'))
#   model.add(Conv2D(64, (3,3), padding='same', name='p2_conv_2'))
#   model.add(MaxPool2D((2,2), padding='same', name='p2_maxpool_1'))

#   model.add(Flatten())
#   model.add(Dense(128, activation='relu', name='p2_dense_1'))
#   model.add(Dense(num_classes, activation='softmax', name='p2_out'))
#   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
  model = keras.models.Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


  return model
