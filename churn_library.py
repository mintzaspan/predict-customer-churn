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
import plotly.express as px


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


def perform_eda(df, num_cols, cat_cols, target_col):
    """Performs EDA on dataframe and save figures to images folder.

    Args:
        df: pandas dataframe
        num_cols: numerical columns list
        cat_cols: categorical columns list
        target_col: target column name

    Returns:
        None
    """
    eda_dir = 'images/eda/'
    if not os.path.exists(eda_dir):
        os.makedirs(eda_dir, exist_ok=True)

    # produce univariate plots
    for i in num_cols:
        fig_num = px.histogram(data_frame=df, x=i, title=i)
        fig_num.write_image(f'{eda_dir}{i}_histplot.png')

    for i in cat_cols:
        fig_cat = px.histogram(data_frame=df, x=i, color=i, title=i)
        fig_cat.write_image(f'{eda_dir}{i}_freqplot.png')

    # produce bivariate plots
    for i in cat_cols:
        avg_target_by_cat = df.groupby(i)[target_col].mean().reset_index()
        fig_bi = px.bar(
            data_frame=avg_target_by_cat,
            x=i,
            y=target_col,
            color=i,
            title=f'Mean {target_col} by {i}')
        fig_bi.write_image(f'{eda_dir}{i}_bv_plot.png')


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

    # define churn variable
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    logging.info('Churn variable created')

    # drop columns
    df.drop(columns=config['drop_columns'], inplace=True)
    logging.info(f"Dropped obsolete columns dropped: {config['drop_columns']}")

    # eda
    logging.info('Starting exploratory data analysis')
    perform_eda(
        df=df,
        num_cols=config["numerical_columns"],
        cat_cols=config["categorical_columns"],
        target_col=config["response"])
    logging.info('Univariate and bivariate analysis plots saved in EDA folder')
