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


def get_paths_angles_tuples(csv_path, minimum_angle):
    """
    Returns a list of (path, steering angle) tuples. Only paths to frames for which
     absolute angles are at least minimum_angle are returned.
    :param csv_path:
    :param minimum_angle:
    :return: (path, angle) tuples list
    """

    with open(csv_path) as file:

        reader = csv.reader(file)
        csv_lines = [line for line in reader]

    path_angle_tuples = []

    for line in csv_lines:

        steering_angle = float(line[3])

        # # Center image
        if abs(steering_angle) >= minimum_angle:

            path_angle_tuples.append((line[0], steering_angle))

        # Left image - only use it if we are not steering fully to the left
        # If we are, then maybe even with camera from left bumper we should steer fully to left
        if steering_angle > -1 and abs(steering_angle) >= minimum_angle:

            modified_angle = np.clip(steering_angle + abs(0.2 * steering_angle), -1, 1)
            path_angle_tuples.append((line[1], modified_angle))

        # # Right image - only use it if we are not steering fully to the right
        # # If we are, then maybe even with camera from right bumper we should steer fully to right
        if steering_angle > -1 and abs(steering_angle) >= minimum_angle:

            modified_angle = np.clip(steering_angle - abs(0.2 * steering_angle), -1, 1)
            path_angle_tuples.append((line[1], modified_angle))

    return path_angle_tuples


def get_single_dataset_generator(csv_path, minimum_angle):
    """
    Return a generator that yields data from a single dataset. On yield return a single (image, steering angle)
    tuple. Image and steering angle are randomly flipped
    :param csv_path: path to drive log
    :param minimum_angle: minimum angle frame must have to be returned
    :return: generator
    """

    paths_angles_tuples = get_paths_angles_tuples(csv_path, minimum_angle)
    parent_dir = os.path.dirname(csv_path)

    while True:

        random.shuffle(paths_angles_tuples)

        for path, steering_angle in paths_angles_tuples:

            # A bit of acrobatics with paths so that we can run generator on AWS (csv_path uses absolute paths)
            relative_path = parent_dir + "/IMG" + path.split("IMG")[1]

            image = cv2.imread(relative_path)

            # Flip randomly
            if random.randint(0, 1) == 1:

                image = cv2.flip(image, flipCode=1)
                steering_angle *= -1

            yield image, steering_angle


def get_multiple_datasets_generator(paths, minimum_angles, batch_size):
    """
    Generator that cycles through multiple datasets
    :param paths: paths to drive log csv files
    :param minimum_angles: minimum angles frame from corresponding data set must have to be outputted
    :param batch_size: batch size
    :return: generator that yields (images, steering_angles) batches.
    """

    # Get generator for each dataset
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

    # Roughly corresponds to 0deg, 5deg and 12.5deg minimum angles
    angles = [0, 0, 0.2, 0.2, 0.5, 0.5]

    training_data_generator = get_multiple_datasets_generator(training_paths, angles, batch_size=128)
    validation_data_generator = get_multiple_datasets_generator(validation_paths, angles, batch_size=128)

    training_samples_count = sum(
        [len(get_paths_angles_tuples(path, minimum_angle)) for path, minimum_angle in zip(training_paths, angles)])

    validation_samples_count = sum(
        [len(get_paths_angles_tuples(path, minimum_angle)) for path, minimum_angle in zip(validation_paths, angles)])

    callbacks = [keras.callbacks.ModelCheckpoint(filepath="./model.h5", verbose=1, save_best_only=True)]

    model = get_model(image_size=(160, 320, 3))
    # model.load_weights("./model.h5")

    history_object = model.fit_generator(
        training_data_generator, samples_per_epoch=training_samples_count, nb_epoch=10,
        validation_data=validation_data_generator, nb_val_samples=validation_samples_count,
        callbacks=callbacks)

    import matplotlib

    # Pyplot import fails with QXcbConnection otherwise...
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    plt.savefig("./loss_plot.png")


if __name__ == "__main__":

    train_model()
