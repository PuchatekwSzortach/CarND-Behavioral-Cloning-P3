"""
Module that logs outputs of different data generators
"""

import os
import logging

import vlogging
import cv2
import numpy as np

import model


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("behavioral_cloning")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def log_single_generator_output(logger):

    generator = model.get_single_dataset_generator(
        "../../data/behavioral_cloning/validation/track_2_curves/driving_log.csv", minimum_angle=0.02)

    for _ in range(10):

        image, steering_angle = next(generator)
        logger.info(vlogging.VisualRecord("Frame", image, "Steering angle: {}".format(steering_angle)))


def log_all_datasets_generator_output(logger):

    parent_dir = "../../data/behavioral_cloning/validation"

    paths = [
        "track_1_center/driving_log.csv",
        "track_2_center/driving_log.csv",
        "track_1_curves/driving_log.csv",
        "track_2_curves/driving_log.csv",
        "track_1_recovery/driving_log.csv",
        "track_2_recovery/driving_log.csv"
    ]

    paths = [os.path.join(parent_dir, path) for path in paths]
    angles = [0, 0, 0.02, 0.02, 0.1, 0.1]

    generator = model.get_multiple_datasets_generator(paths, angles, batch_size=6)

    for _ in range(10):

        images, steering_angles = next(generator)

        # Resize images slightly
        images = [cv2.pyrDown(image) for image in images]

        logger.info(vlogging.VisualRecord(
            "Frames", images, "Steering angle: {}".format(steering_angles.tolist())))


def log_preprocessed_datasets_generator_output(logger):

    parent_dir = "../../data/behavioral_cloning/validation/"

    paths = [
        "track_1_center/driving_log.csv",
        "track_2_center/driving_log.csv",
        "track_1_curves/driving_log.csv",
        "track_2_curves/driving_log.csv",
        "track_1_recovery/driving_log.csv",
        "track_2_recovery/driving_log.csv"
    ]

    paths = [os.path.join(parent_dir, path) for path in paths]
    angles = [0, 0, 0.02, 0.02, 0.1, 0.1]

    generator = model.get_multiple_datasets_generator(paths, angles, batch_size=6)
    preprocessing_model = model.get_preprocessing_model(image_size=(160, 320, 3))

    for _ in range(10):

        images, steering_angles = next(generator)

        preprocessed_images = preprocessing_model.predict(images)
        preprocessed_images = [(255 * image).astype(np.uint8) for image in preprocessed_images]

        logger.info(vlogging.VisualRecord(
            "Frames", preprocessed_images, "Steering angle: {}".format(steering_angles.tolist())))


def main():

    logger = get_logger("/tmp/behavioral_cloning.html")

    # log_single_generator_output(logger)
    log_all_datasets_generator_output(logger)
    # log_preprocessed_datasets_generator_output(logger)


if __name__ == "__main__":

    main()