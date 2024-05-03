from churn_library import import_data, perform_eda, split_frame, train_model, load_model
import logging
import os


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

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: SUCCESS - The file appears to have rows and columns")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_eda(df):
    num_cols = ['num_col1', 'num_col2']
    cat_cols = ['cat_col1', 'cat_col2']
    target_col = 'target_col'
    try:
        perform_eda(df, num_cols, cat_cols, target_col)
        logging.info("Testing perform_eda: SUCCESS - The EDA was performed")
    except Exception as err:
        logging.error("Testing perform_eda: The EDA wasn't performed")
        raise err

    # Check if images are created
    for col in num_cols:
        assert os.path.exists(f'images/eda/{col}_histplot.png')
    for col in cat_cols:
        assert os.path.exists(f'images/eda/{col}_freqplot.png')
        assert os.path.exists(f'images/eda/{col}_bv_plot.png')
    logging.info("Testing perform_eda: SUCCESS - The EDA images were created")


def test_split_frame(df, target_col='target_col', test_size=0.2):
    """Test the split_frame function.

    Args:
        df (pd.DataFrame): The dataframe to split.

    Returns:
        None
    """
    try:
        X_train, X_test, y_train, y_test = split_frame(
            df, target_col, test_size)
    except Exception as err:
        logging.error("Testing split_frame: The data wasn't split")
        raise err
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
    logging.info(
        "Testing split_frame: SUCCESS - The data was split successfully")


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

    try:
        train_model(
            algo=algo,
            X=X_train,
            y=y_train,
            num_cols=num_cols,
            cat_cols=cat_cols,
            params=config['random_forest']['param_grid'])
        logging.info(
            "Testing train_model: SUCCESS - Model with parameters for grid search trained successfully")
    except Exception as err:
        logging.error(
            "Testing train_model: Model with parameters for grid search wasn't trained")
        raise err

    assert os.path.exists(f'models/{algo}.pkl')
    logging.info("Testing train_model: SUCCESS - Model was saved successfully")

    other_algo = "logistic_regression"
    try:
        train_model(
            algo=other_algo,
            X=X_train,
            y=y_train,
            num_cols=num_cols,
            cat_cols=cat_cols)
        logging.info(
            "Testing train_model: SUCCESS - Model without grid search parameters trained successfully")
    except Exception as err:
        logging.error(
            "Testing train_model: Model without grid search parameters wasn't trained")
        raise err

    assert os.path.exists(f'models/{other_algo}.pkl')
    logging.info("Testing train_model: SUCCESS - Model was saved successfully")


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

    try:
        model = load_model(f"models/{algo}.pkl")
        logging.info(
            "Testing load_model: SUCCESS - Model was loaded successfully")
    except Exception as err:
        logging.error("Testing load_model: Model wasn't loaded")
        raise err

    assert model is not None
    logging.info("Testing load_model: SUCCESS - Model type is not None")


if __name__ == "__main__":
    pass
