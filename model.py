"""
Module with preprocessing, prediction model and training code
"""

import os
import csv
import random
import itertools

import keras
import numpy as np
import tensorflow as tf
import cv2


def get_preprocessing_pipeline(x):
    """
    Crops, resizes and scales input data
    """

    x = keras.layers.Cropping2D(cropping=((50, 20), (0, 0)))(x)
    x = keras.layers.Lambda(lambda data: tf.image.resize_images(data, size=(45, 160)))(x)
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


def get_model(image_size):

    expected_image_size = (160, 320, 3)

    if image_size != expected_image_size:
        raise ValueError("Expected image size is {}, but {} was given".format(expected_image_size, image_size))

    input = keras.layers.Input(shape=image_size)
    x = get_preprocessing_pipeline(input)

    model = keras.models.Model(input=input, output=x)
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

    while True:

        random.shuffle(csv_lines)

        for line in csv_lines:

            center_image = line[0]
            steering_angle = float(line[3])

            image = cv2.imread(center_image)

            # Flip randomly
            if random.randint(0, 1) == 1:

                image = cv2.flip(image, flipCode=1)
                steering_angle *= -1

            yield image, steering_angle


def get_multiple_datasets_generator(paths, minimum_angles, batch_size):

    generators = [get_single_dataset_generator(path, minimum_angle)
                  for path, minimum_angle in zip(paths, minimum_angles)]

    generators_cycle = itertools.cycle(generators)

    while True:

        images = []
        steering_angles = []

        while len(images) < batch_size:

            current_generator = next(generators_cycle)
            image, angle = next(current_generator)

            images.append(image)
            steering_angles.append(angle)

        yield images, steering_angles


