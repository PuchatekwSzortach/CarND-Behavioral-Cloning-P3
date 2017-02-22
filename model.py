"""
Module with preprocessing, prediction model and training code
"""

import os
import csv
import random
import itertools

import numpy as np
import cv2
import keras


def get_preprocessing_pipeline(x):
    """
    Crops, resizes and scales input data
    """

    x = keras.layers.Cropping2D(cropping=((50, 20), (0, 0)))(x)

    # Poor man's resize, since using
    # x = keras.layers.Lambda(lambda data: tf.image.resize_images(data, size=(45, 160)))(x)
    # fails on model load...
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Lambda(lambda data: data / 255)(x)

    return x


def get_preprocessing_model(image_size):
    """
    Return model that does only data preprocessing
    """

    expected_image_size = (160, 320, 3)

    if image_size != expected_image_size:
        raise ValueError("Expected image size is {}, but {} was given".format(expected_image_size, image_size))

    input = keras.layers.Input(shape=image_size)
    x = get_preprocessing_pipeline(input)

    model = keras.models.Model(input=input, output=x)
    return model


def get_prediction_pipeline(x):

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(p=0.5)(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(p=0.5)(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='elu', border_mode='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(p=0.5)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(output_dim=1)(x)

    return x


def get_model(image_size):

    expected_image_size = (160, 320, 3)

    if image_size != expected_image_size:
        raise ValueError("Expected image size is {}, but {} was given".format(expected_image_size, image_size))

    input = keras.layers.Input(shape=image_size)
    x = get_preprocessing_pipeline(input)
    x = get_prediction_pipeline(x)

    model = keras.models.Model(input=input, output=x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


class VideoProcessor:
    """
    A simple class that can be used together with moviepy to see how our preprocessing pipeline affects images
    """

    def __init__(self):

        self.model = get_preprocessing_model(image_size=(160, 320, 3))

    def process_frame(self, frame):

        processed_frame = self.model.predict(np.array([frame]))[0]
        return 255 * processed_frame


def get_single_dataset_generator(csv_path, minimum_angle=0):
    """
    Return a generator that yields data from a single dataset. Single yield return a single (image, steering angle)
    tuple. Image and steering angle are randomly flipped
    :param csv_path: path to drive log
    :param minimum_angle: minimum angle frame must have to be returned
    :return: generator
    """

    with open(csv_path) as file:

        reader = csv.reader(file)
        csv_lines = [line for line in reader if abs(float(line[3])) >= minimum_angle]

    parent_dir = os.path.dirname(csv_path)

    while True:

        random.shuffle(csv_lines)

        for line in csv_lines:

            # A bit of acrobatics with paths so that we can run generator on AWS (csv_path uses absolute paths)
            path = parent_dir + "/IMG" + line[0].split("IMG")[1]

            steering_angle = float(line[3])

            image = cv2.imread(path)

            # Flip randomly
            if random.randint(0, 1) == 1:

                image = cv2.flip(image, flipCode=1)
                steering_angle *= -1

            yield image, steering_angle


def get_dataset_samples_count(csv_path, minimum_angle):

    with open(csv_path) as file:

        reader = csv.reader(file)
        csv_lines = [line for line in reader if abs(float(line[3])) >= minimum_angle]

    return len(csv_lines)


def get_multiple_datasets_generator(paths, minimum_angles, batch_size):
    """
    Generator that cycles through multiple datasets
    :param paths: paths to drive log csv files
    :param minimum_angles: minimum angles frame from corresponding data set must have to be outputted
    :param batch_size: batch size
    :return: generator that yields images, steering_angles batches.
    """

    generators = [get_single_dataset_generator(path, minimum_angle)
                  for path, minimum_angle in zip(paths, minimum_angles)]

    # We will cycle through different generators, so that each dataset is sampled evenly
    generators_cycle = itertools.cycle(generators)

    while True:

        images = []
        steering_angles = []

        while len(images) < batch_size:

            current_generator = next(generators_cycle)
            image, angle = next(current_generator)

            images.append(image)
            steering_angles.append(angle)

        yield np.array(images), np.array(steering_angles)


def train_model():

    training_parent_dir = "../../data/behavioral_cloning/training/"
    validation_parent_dir = "../../data/behavioral_cloning/validation/"

    paths = [
        "track_1_center/driving_log.csv",
        "track_2_center/driving_log.csv",
        "track_1_curves/driving_log.csv",
        "track_2_curves/driving_log.csv",
        "track_1_recovery/driving_log.csv",
        "track_2_recovery/driving_log.csv"
    ]

    training_paths = [os.path.join(training_parent_dir, path) for path in paths]
    validation_paths = [os.path.join(validation_parent_dir, path) for path in paths]

    angles = [0, 0, 0.02, 0.02, 0.1, 0.1]

    trainig_data_generator = get_multiple_datasets_generator(training_paths, angles, batch_size=128)
    validation_data_generator = get_multiple_datasets_generator(validation_paths, angles, batch_size=128)

    training_samples_count = sum(
        [get_dataset_samples_count(path, minimum_angle) for path, minimum_angle in zip(training_paths, angles)])

    validation_samples_count = sum(
        [get_dataset_samples_count(path, minimum_angle) for path, minimum_angle in zip(validation_paths, angles)])

    callbacks = [keras.callbacks.ModelCheckpoint(filepath="./model.h5", verbose=1, save_best_only=True)]

    model = get_model(image_size=(160, 320, 3))
    # model.load_weights("./model.h5")

    model.fit_generator(trainig_data_generator, samples_per_epoch=training_samples_count, nb_epoch=10,
                        validation_data=validation_data_generator, nb_val_samples=validation_samples_count,
                        callbacks=callbacks)


if __name__ == "__main__":

    train_model()
