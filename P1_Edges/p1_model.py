from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from keras.models import Sequential

# model.add(Input(shape=(256,256,3)))

model = Sequential()
# Add convolutional layers to extract edges
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))


# Add upsampling layers to reconstruct edge map
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')