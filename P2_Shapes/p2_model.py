import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras.optimizers import Adam, SGD 
from keras import regularizers
from keras.models import Sequential

def build_model(num_classes, args):
      
  if args.environment == 'hpc':
        P1_MODEL_PATH = "/home/niranjan.rajesh_ug23/DCV/models/best_p1/my_model.h5"
  elif args.environment == 'local':
        P1_MODEL_PATH = "C:/Niranjan/Ashoka/Research/DCV/Models/P1_Edges3.0/my_model.h5"    

  if not args.vanilla:
      model = tf.keras.models.load_model(P1_MODEL_PATH)

      # removing the deconv and upsampling layers
      model = tf.keras.models.Sequential(model.layers[:-5])
      
      if args.freeze_weights:
          for layer in model.layers:
                layer.trainable = False

      model.add(Conv2D(32, (3,3), padding='same', name='p2_conv_1', activation='relu'))
      model.add(Conv2D(64, (3,3), padding='same', name='p2_conv_2', activation='relu'))
      model.add(MaxPooling2D((2,2), padding='same', name='p2_maxpool_1'))

      model.add(Conv2D(128, (3,3), padding='same', name='p2_conv_3', activation='relu'))
      model.add(Conv2D(128, (3,3), padding='same', name='p2_conv_4', activation='relu'))
      model.add(MaxPooling2D((2,2), padding='same', name='p2_maxpool_2'))

      model.add(Flatten())
      model.add(Dense(512, activation='relu', name='p2_dense_1'))
      model.add(Dropout(rate=0.4))
      model.add(Dense(256, activation='relu', name='p2_dense_2'))
      model.add(Dropout(rate=0.4))

      model.add(Dense(num_classes, activation='softmax', name='p2_out'))
      
      opt = Adam(lr=args.learning_rate)
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
  elif args.vanilla:
      model = Sequential()
      # Add convolutional layers to extract edges
      model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3), kernel_regularizer=regularizers.l2(0.01)))
      model.add(Dropout(0.5))
      model.add(MaxPooling2D((2, 2), padding='same'))
      
      model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
      model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
      model.add(MaxPooling2D((2,2), padding='same'))

      model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
      model.add(Conv2D(128, (3,3), padding='same',  activation='relu'))
      model.add(MaxPooling2D((2,2), padding='same'))

      model.add(Flatten())
      model.add(Dense(512, activation='relu'))
      model.add(Dropout(rate=0.4))
      model.add(Dense(256, activation='relu'))
      model.add(Dropout(rate=0.4))

      model.add(Dense(num_classes, activation='softmax'))
      
      
      opt = Adam(lr=args.learning_rate)
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


  return model
