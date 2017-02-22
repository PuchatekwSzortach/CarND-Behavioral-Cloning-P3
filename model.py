"""
Module with preprocessing, prediction model and training code
"""

import os
import csv

import keras
import numpy as np
import tensorflow as tf


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


def get_single_dataset_generator(csv_path):

    csv_lines = []

    with open(csv_path) as file:

        reader = csv.reader(file)

        for line in reader:

            csv_lines.append(line)

    while True:

        for line in csv_lines:

            yield(line)