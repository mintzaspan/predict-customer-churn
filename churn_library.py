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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import dill


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


def split_frame(df, response, test_size):
    """Splits pandas DataFrame to train and test datasets

    Args:
        df: pandas DataFrame
        response: string of response column name

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """

    y = df[response]
    X = df.drop(columns=[response])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)
    return (X_train, X_test, y_train, y_test)


def train_model(
        algo,
        X,
        y,
        num_cols,
        cat_cols,
        params=None
):
    """Trains a binary classifier and saves as pickle object

    Args:
        algo: string for selected binary classification algorithm i.e. "logistic regression" or "random forest"
        X: predictors
        y: response
        num_cols: list of numerical type columns in X
        cat_cols: list of categorical type columns in X

    Returns:
        pipe : binary classification pipeline
    """
    # create models dir
    models_dir = 'models/'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    # preprocessor
    num_features = num_cols
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_features = cat_cols
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", TargetEncoder()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # algorithms
    algo_dict = {
        "logistic_regression": LogisticRegression(random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(seed=0),
        "support_vector": SVC(random_state=42)
    }

    selected_algo = algo_dict[algo]

    # pipeline
    if params is None:
        clf = selected_algo
    else:
        clf = GridSearchCV(
            estimator=selected_algo,
            param_grid=params,
            cv=3,
            n_jobs=-1,
            refit=True
        )

    # fit
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ]
    )

    pipe.fit(X, y)

    # save to local folder
    with open(f'{models_dir}{algo}.pkl', 'wb') as f:
        dill.dump(pipe, f)

    return (pipe)


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

    # split to train test
    X_train, X_test, y_train, y_test = split_frame(
        df=df, response=config["response"], test_size=config["feature_engineering"]["test_size"])
    logging.info(
        "Dataframe was split to X_train, X_test, y_train, y_test frames")

    # train random forest and logistic regression
    for i in ["logistic_regression", "random_forest"]:
        if i in config:
            train_model(
                algo=i,
                X=X_train.iloc[:1000],
                y=y_train[:1000],
                num_cols=config["numerical_columns"],
                cat_cols=config["categorical_columns"],
                params=config[i]["param_grid"]
            )
        else:
            train_model(
                algo=i,
                X=X_train.iloc[:1000],
                y=y_train[:1000],
                num_cols=config["numerical_columns"],
                cat_cols=config["categorical_columns"],
                params=None
            )
        logging.info(
            f"{i} model trained")
        logging.info("Model saved in models folder")
