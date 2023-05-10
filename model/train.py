# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adadelta
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import os

img_width, img_height = 47, 47
data_dir = 'data'
batch_size = 16
epochs = 100

train_data, validation_data = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale'
)

class_names = train_data.class_names
print(class_names)

nb_train_samples = len(train_data)
nb_validation_samples = len(validation_data)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(13))
model.add(Activation('softmax'))

model.compile(optimizer=Adadelta(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['sparse_categorical_accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor the validation loss
    min_delta=0,           # Minimum change in the monitored metric to qualify as an improvement
    patience=5,            # Number of epochs with no improvement after which training will be stopped
    verbose=1,             # Print a message when training is stopped
    mode='min'             # In 'min' mode, training will stop when the monitored metric stops decreasing
)

model.fit(train_data,
    epochs = epochs,
    validation_data = validation_data,
    callbacks=[early_stopping]
)

model.save('model_saved.h5')