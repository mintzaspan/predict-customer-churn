import os
import logging
import pytest
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """Test the import_data function
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture
def df():
    data = pd.DataFrame({
        'num_col1': [1, 2, 3, 4, 5],
        'num_col2': [5, 4, 3, 2, 1],
        'cat_col1': ['a', 'b', 'a', 'b', 'a'],
        'cat_col2': ['b', 'a', 'b', 'a', 'b'],
        'target_col': [1, 0, 1, 0, 1]
    })
    return data


def test_perform_eda(df):
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    target_col = 'target_col'
    perform_eda(df, num_cols, cat_cols, target_col)

    # Check if images are created
    for col in num_cols + cat_cols:
        assert os.path.exists(f'images/eda/{col}_histplot.png')
        assert os.path.exists(f'images/eda/{col}_freqplot.png')
        assert os.path.exists(f'images/eda/{col}_bv_plot.png')


def test_split_frame(df):
    X_train, X_test, y_train, y_test = split_frame(df, 'target_col')
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
