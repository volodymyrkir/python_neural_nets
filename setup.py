"""This module is used for common functionality"""
import os
from consts import DATA_DIR


def get_task_data(file_name: str) -> str:
    """
    Returns path to input data file.

    :param file_name: Name of the file (with format)
    :return: path to file.
    """
    return os.path.join(os.getcwd(), os.pardir, DATA_DIR, file_name)