from churn_library import import_data, perform_eda, split_frame, train_model, load_model
import logging
import os


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


def test_split_frame(df, target_col='target_col', test_size=0.2):
    """Test the split_frame function.

    Args:
        df (pd.DataFrame): The dataframe to split.

    Returns:
        None
    """
    X_train, X_test, y_train, y_test = split_frame(
        df, target_col, test_size)
    assert X_train.shape[0] > 0
    assert X_train.shape[1] > 0
    assert X_test.shape[0] > 0
    assert X_test.shape[1] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == df.shape[0]


def test_train_model(df, config):
    """Test the train_model function.

    Args:
        df (pd.DataFrame): The dataframe to split and train the model on.

    Returns:
        None
    """
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    target_col = 'target_col'
    algo = 'random_forest'
    X_train, X_test, y_train, y_test = split_frame(
        df, target_col, test_size=0.2)
    train_model(
        algo=algo,
        X=X_train,
        y=y_train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        params=config['random_forest']['param_grid'])
    assert os.path.exists(f'models/{algo}.pkl')
    other_algo = "logistic_regression"
    train_model(
        algo=other_algo,
        X=X_train,
        y=y_train,
        num_cols=num_cols,
        cat_cols=cat_cols)
    assert os.path.exists(f'models/{other_algo}.pkl')


def test_load_model(df):
    """Test the load_model function.

    Args:
        df (pd.DataFrame): The dataframe to split and train the model on.

    Returns:
        None
    """
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    target_col = 'target_col'
    algo = 'logistic_regression'
    X_train, X_test, y_train, y_test = split_frame(
        df, target_col, test_size=0.2)
    train_model(
        algo=algo,
        X=X_train,
        y=y_train,
        num_cols=num_cols,
        cat_cols=cat_cols)
    model = load_model(f"models/{algo}.pkl")
    assert model is not None


if __name__ == "__main__":
    pass
