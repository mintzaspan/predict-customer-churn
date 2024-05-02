"""
Purpose: Tests the functions in the churn_library.py script and logs results.
Author: Panagiotis Mintzas
Date: May 2024
"""

import os
import logging
import pytest
import tempfile
from churn_library import *
import yaml

logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def config():
    """
    Load the configuration file.

    Args:
        None

    Returns:
        config (dict): The configuration file as a dictionary.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        logging.info("Configuration file loaded")
    return config


@pytest.fixture(scope='module')
def test_import(config):
    """
    Test the import_data function.

    This function tests the import_data function by creating a temporary CSV file,
    importing the data from the file, and performing assertions on the imported data.

    Args:
        data (str): The path to the data file.

    Returns:
        None (or error messages if the assertions fail)
    """

    try:
        df = import_data(config['data'])
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: The file appears to have rows and columns")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope='module')
def df():
    """
    Creates a sample dataframe for testing purposes.

    Args:
        None

    Returns:
        data (pd.DataFrame): A sample dataframe for running tests.
    """
    data = pd.DataFrame({
        'num_col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'num_col2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        'cat_col1': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a'],
        'cat_col2': ['b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
        'target_col': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return data


def test_perform_eda(df):
    """
    Test the perform_eda function.

    Args:
        df (pd.DataFrame): A sample dataframe for running tests.

    Returns:
    """
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    target_col = 'target_col'
    perform_eda(df, num_cols, cat_cols, target_col)

    # Check if images are created
    for col in num_cols:
        assert os.path.exists(f'images/eda/{col}_histplot.png')
    for col in cat_cols:
        assert os.path.exists(f'images/eda/{col}_freqplot.png')
        assert os.path.exists(f'images/eda/{col}_bv_plot.png')


def test_split_frame(df):
    X_train, X_test, y_train, y_test = split_frame(
        df=df, response='target_col', test_size=0.2)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0]


def test_train_model(df):
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    algo = 'logistic_regression'
    X_train, X_test, y_train, y_test = split_frame(df, 'target_col', 0.2)
    model = train_model(algo, X_train, y_train, num_cols, cat_cols)
    assert model is not None
    assert os.path.exists('models/logistic_regression.pkl')


def test_load_model():
    model = load_model('models/logistic_regression.pkl')
    assert model is not None


def test_build_classification_report(df):
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    algo = 'logistic_regression'
    X_train, X_test, y_train, y_test = split_frame(df, 'target_col', 0.2)
    model = train_model(algo, X_train, y_train, num_cols, cat_cols)
    build_classification_report(
        models=[model],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test)
    assert os.path.exists(
        'images/results/LogisticRegression_classification_report.png')
    assert os.path.exists('images/results/ROC_AUC.png')


def test_get_features_importance(df):
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    algo = 'logistic_regression'
    X_train, X_test, y_train, y_test = split_frame(df, 'target_col', 0.2)
    model = train_model(algo, X_train, y_train, num_cols, cat_cols)
    get_feature_importances(models=[model], X=X_train)
    assert os.path.exists(
        'images/results/LogisticRegression feature importance plot.png')
