import tensorflow as tf


P1_MODEL_PATH = "C:/Niranjan/Ashoka/Research/DCV/Models/P1_Edges1.0/my_model.h5"

model = tf.keras.models.load_model(P1_MODEL_PATH)

model.summary()