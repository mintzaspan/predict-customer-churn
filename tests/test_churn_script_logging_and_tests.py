import os
import logging
from churn_library import import_data, perform_eda


# Set up logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import_data(data_pth):
    """Test the import_data function.

    Args:
        data_pth (str): The path to the data directory.

    Returns:
        None

    """

    try:
        df = import_data(data_pth)
        logging.info("Testing import_data: SUCCESS - The file was found")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
    logging.info(
        "Testing import_data: SUCCESS - The file appears to have rows and columns")


def test_perform_eda(df):
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


if __name__ == "__main__":
    pass
