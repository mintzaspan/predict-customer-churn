import pytest
import yaml


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
