from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras import regularizers

# model.add(Input(shape=(256,256,3)))

model = Sequential()
# Add convolutional layers to extract edges
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3), name='p1_conv_1', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2), padding='same', name='p1_maxpool_1'))


# Add upsampling layers to reconstruct edge map. These will be removed for P2 training.
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
opt = SGD(lr=1e-3)
model.compile(optimizer=opt, loss='binary_crossentropy')