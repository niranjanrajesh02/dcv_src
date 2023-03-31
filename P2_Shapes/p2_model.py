import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input


P1_MODEL_PATH = "C:/Niranjan/Ashoka/Research/DCV/Models/P1_Edges3.0/my_model.h5"

def build_model(num_classes):
  model = tf.keras.models.load_model(P1_MODEL_PATH)

  model = tf.keras.models.Sequential(model.layers[:-5])
  
  for layer in model.layers:
    layer.trainable = False

  model.add(Conv2D(32, (3,3), padding='same', name='p2_conv_1'))
  model.add(MaxPool2D((2,2), padding='same', name='p2_maxpool_1'))

  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model
