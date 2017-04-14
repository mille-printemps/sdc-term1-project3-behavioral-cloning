# coding : utf-8

import csv
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

LOG_PATH = 'data/driving_log.csv'
IMAGE_PATH = 'data/IMG'
STEERING_CORRECTION = 0.2
EPOCHS = 7

def generator(data, batch_size=32):
    n_data = len(data)
    while 1:
        shuffle(data)
        for offset in range(0, n_data, batch_size):
            batches = data[offset:offset+batch_size]

            images = []
            measurements = []
            for batch in batches:
                image_center = cv2.imread(os.path.join(IMAGE_PATH, batch[0].split('/')[-1]))
                image_left = cv2.imread(os.path.join(IMAGE_PATH, batch[1].split('/')[-1]))
                image_right = cv2.imread(os.path.join(IMAGE_PATH, batch[2].split('/')[-1]))

                steering_center = float(batch[3])
                steering_left = steering_center + STEERING_CORRECTION
                steering_right = steering_center - STEERING_CORRECTION

                images.extend([image_center, image_left, image_right])                
                measurements.extend([steering_center, steering_left, steering_right])

                images.append(cv2.flip(image_center, 1))
                measurements.append(steering_center * -1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

lines = []
with open(LOG_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
            
train_data, validation_data = train_test_split(lines, test_size=0.2)
train_generator = generator(train_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)
    
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
metrics = model.fit_generator(train_generator, samples_per_epoch=len(train_data),\
                              validation_data=validation_generator, nb_val_samples=len(validation_data),\
                              verbose=1, nb_epoch=EPOCHS)
model.save('model.h5')

print(metrics.history.keys())
plt.plot(metrics.history['loss'])
plt.plot(metrics.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
