"""
Module that logs outputs of different data generators
"""

import os
import logging

import vlogging

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


def main():

    logger = get_logger("/tmp/behavioral_cloning.html")

    generator = model.get_single_dataset_generator("../../data/behavioral_cloning/track_1_center/driving_log.csv")

    for _ in range(10):

        print(next(generator))


if __name__ == "__main__":

    main()