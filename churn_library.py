"""
Purpose: Build a model to predict customer churn
Author: Panagiotis Mintzas
Date: March 2024
"""


# import libraries
import os
import logging
import yaml
import pandas as pd


def setup_logging(script_pth):
    """Sets up logging for the current module named after the module

    Args:
        script_pth: path to a python module

    Returns:
        None
    """

    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    script_path = script_pth
    script_name = os.path.basename(script_path).split('/')[-1].split('.')[0]
    log_name = ''.join([script_name, '.log'])
    log_file = os.path.join(log_dir, log_name)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    """Returns pandas dataframe for the CSV found at pth

    Args:
        pth: a path to the csv

    Returns:
        df: pandas dataframe
    """
    data = pd.read_csv(pth)
    return (data)


if __name__ == "__main__":

    # set up logging
    setup_logging('churn_library.py')
    logging.info("Log file created")

    # read config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Training parameters imported")

    # import data
    try:
        logging.info('Reading CSV file')
        df = import_data(config['data'])
        logging.info('SUCCESS: File read into dataframe')
    except FileNotFoundError:
        logging.error('File was not found in specified directory')
