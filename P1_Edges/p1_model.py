from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from keras.models import Sequential

# model.add(Input(shape=(256,256,3)))

model = Sequential()
# Add convolutional layers to extract edges
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3), name='p1_conv_1'))
model.add(MaxPooling2D((2, 2), padding='same', name='p1_maxpool_1'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='p1_conv_2'))
model.add(MaxPooling2D((2, 2), padding='same', name='p1_maxpool_2'))


# Add upsampling layers to reconstruct edge map
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')