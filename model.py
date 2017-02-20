import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random as rand
import json

tf.python.control_flow_ops = tf

# Preparing path for reading data
rootpath = './Dataset'
filename = 'driving_log.csv'
csvdatafile = rootpath + '/' + filename

# size of the image as expected by the model
img_ch, img_rows, img_cols = 3, 66, 200


# implementation of Nvidia model
def get_nvidia_model():

    """Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf"""

    model = Sequential()
    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x / 127.5 - 1.,
        input_shape=(img_rows, img_cols, img_ch),
        output_shape=(img_rows, img_cols, img_ch))
    )
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), name='conv1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), name='conv2'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), name='conv3'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv4'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv5'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, name='fc2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, name='fc3'))
    model.add(Activation('relu'))
    model.add(Dense(1, name='out'))
    model.add(Activation('linear'))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model

train_images = []
train_angles = []
raw_data = []
# steering angle correction offset to apply to left and right camera images
steering_correction = 0.15

# read the csv file
with open(csvdatafile) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for line in reader:
        raw_data.append(line)

# split the data read from csv into training and validation datasets
train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=23)

# For every image in training data, add left and right camera images with angle correction
# Here only the image path is used. Reading the image is handled in the generator
for index in range(0, len(train_data)):
    row = train_data[index]
    for col_index in range(0, 3):
        img_path = row[col_index]
        tokens = img_path.split("\\")
        virtual_path = tokens[-2] + '/' + tokens[-1]
        train_images.append(virtual_path)
    center_angle = row[3]
    train_angles.append(float(center_angle))
    train_angles.append(float(center_angle) + steering_correction)
    train_angles.append(float(center_angle) - steering_correction)


# Brightness augmentation
def image_augment(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # augment the Y-channel (intensity)
    img_yuv[:, :, 0] = img_yuv[:, :, 0] * (0.2 + np.random.uniform())  # 0.5
    img_aug = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_aug


# Adding horizontal and vertical shifts
def image_shift(image, steer):
    # Translation
    tr_x = 100*np.random.uniform()-50
    tr_y = 20*np.random.uniform()-10
    translation_matrix = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, translation_matrix, (img_cols, img_rows))
    steer_ang = steer + tr_x * 0.01
    return image_tr, steer_ang


# training generator
def train_generator(train_img, train_ang, batch_size=128):
    num_samples = len(train_img)
    # Only half of the batch given as a parameter is used. The yielded batch will be then doubled after flipping
    local_batch_size = int(batch_size / 2)
    while True:
        train_img, train_ang = shuffle(train_img, train_ang)
        for offset in range(0, num_samples, local_batch_size):
            batch_images = train_img[offset:offset + local_batch_size]
            batch_angles = train_ang[offset:offset + local_batch_size]
            images = []
            angles = []
            for image, angle in zip(batch_images, batch_angles):
                raw_image = plt.imread(rootpath + '/' + image)
                angle = float(angle)
                cropped_image = raw_image[70:135, :]
                resized_image = cv2.resize(cropped_image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
                shifted_image, angle = image_shift(resized_image, angle)
                augmented_image = image_augment(shifted_image)
                flipped_image = cv2.flip(augmented_image, 1)
                flipped_angle = -angle
                images.append(augmented_image)
                angles.append(angle)
                images.append(flipped_image)
                angles.append(flipped_angle)
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


# Validation generator. Images are only cropped and resized
def val_generator(validation_data, batch_size=128):
    num_samples = len(validation_data)
    while True:
        rand.shuffle(validation_data)
        for offset in range(0, num_samples, batch_size):
            val_batches = validation_data[offset:offset + batch_size]
            images = []
            angles = []
            for batch in val_batches:
                val_tokens = batch[0].split("\\")
                path = val_tokens[-2] + '/' + val_tokens[-1]
                raw_image = plt.imread(rootpath + '/' + path)
                angle = float(batch[3])
                cropped_image = raw_image[70:135, :]
                resized_image = cv2.resize(cropped_image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
                images.append(resized_image)
                angles.append(angle)
            x_val = np.array(images)
            y_val = np.array(angles)
            yield shuffle(x_val, y_val)


# Model training
model = get_nvidia_model()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator(train_images, train_angles, batch_size=128),
                    samples_per_epoch=20000,
                    validation_data=val_generator(val_data, batch_size=128),
                    verbose=1,
                    nb_val_samples=len(val_data),
                    nb_epoch=5)


# Saving the trained model
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
model.save('./model.h5')
print("Model Saved")
