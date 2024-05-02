import pytest
import yaml
import pandas as pd


@pytest.fixture(scope="session")
def config():
    """Load the config.yaml file and return it as a dictionary.

    Args:
        None

    Returns:
        dict: The contents of the config.yaml file as a dictionary.

    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def data_pth(config):
    """Return the path to the data directory.

    Args:
        config (dict): The contents of the config.yaml file as a dictionary.

    Returns:
        str: The path to the data directory.

    """
    return config["data"]


@pytest.fixture(scope="session")
def df():
    """Return a dummy DataFrame with five columns for testing. Two columns are numerical, two are categorical and one is the 'target' column.

    Args:
        None

    Returns:
        pandas.DataFrame: A dummy DataFrame.

    """

    data = pd.DataFrame(
        {
            'num_col1': [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20],
            'num_col2': [
                5,
                4,
                3,
                2,
                1,
                5,
                4,
                3,
                2,
                1,
                5,
                4,
                3,
                2,
                1,
                5,
                4,
                3,
                2,
                1],
            'cat_col1': [
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b'],
            'cat_col2': [
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a',
                'b',
                'a'],
            'target_col': [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0]})

    return data
