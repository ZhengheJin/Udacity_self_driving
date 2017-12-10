import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os.path

lines = []
# Read in the training data from driving counter-clock wise
with open('data/forward/driving_log.csv') as csvfile:
#with open('/home/patrick/Downloads/RuiMao/CarND-Behavioral-Cloning-P3-master/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

# Read in the training data from driving clock wise (reverse of countern
# -clock wise)
with open('data/reverse/driving_log.csv') as csvfile2:
#with open('../../../recordings_reverse_track_try2/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line2 in reader2:
        lines.append(line2)
print(len(lines))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# Define the generator, so that the images can be read in batches, to save memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                # Need to use \, as windows path are separated by \ instead of /
                filename = source_path.split('/')[-1]
                if os.path.isfile('data/forward/IMG/' + filename):
                    name = 'data/forward/IMG/' + filename
                else:
                    name = 'data/reverse/IMG/' + filename
                #print("DEBUG:"+name)

                center_image = cv2.imread(name,1)
                image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # print(image.shape)
                images.append(image_rgb)
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                # Augment the image by filpping it
                images.append(cv2.flip(image_rgb, 1))
                angles.append(center_angle * -1.0)


            X_train = np.array(images)
            y_train = np.array(angles)
            yield (shuffle(X_train, y_train))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

'''
# First iteration of the code
# Reads in all the training data at once
images = []
measurements = []
for i, line in enumerate(lines):
    #print(i)
    source_path = line[0]
    # Need to use \, as windows path are separated by \ instead of /
    filename = source_path.split('\\')[-1]
    #print(filename)
    current_path = '../../../recordings/IMG/' + filename
    image = cv2.imread(current_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    images.append(image_rgb)
    measurement = float(line[3])
    measurements.append(measurement)


aug_images, aug_measures = [], []
for img, measure in zip(images, measurements):
    aug_images.append(img)
    aug_measures.append(measure)
    aug_images.append(cv2.flip(img, 1))
    aug_measures.append(measure * -1.0)

#X_train = np.array(images)
#y_train = np.array(measurements)

X_train = np.array(aug_images)
y_train = np.array(aug_measures)
'''


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

'''
# Model NO. 1 -- Simplest model without CNN
model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
'''

'''
# Model NO. 2 -- Google Lenet
model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
# Crop the top 60 pixels and bottom 20 pixels
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

# Model NO. 3 -- Nvidia CNN
# This architecture offers the best result for this project
model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
# Crop the top 60 pixels and bottom 20 pixels
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# Add dropout to prevent overfitting
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
