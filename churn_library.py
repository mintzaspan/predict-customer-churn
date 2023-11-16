# library doc string


# import libraries
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info('Reading csv file')
        df = pd.read_csv(pth)
        logging.info('SUCCESS: File read into dataframe')
        return (df)
    except FileNotFoundError:
        print('File was not found in the specified directory')
        logging.error('File was not found in specified directory')


def perform_eda(df, num_cols, cat_cols, target_col):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    logging.info('Starting exploratory data analysis')
    # df shape
    df_shape = df.shape
    logging.info(f"Dataframe has {df_shape[0]} rows and {df_shape[1]} columns.")

    # null counts
    missing_values = df.isnull().sum()
    high_missing_thresh = 0.3
    high_missing_rate = missing_values[missing_values>len(df)*high_missing_thresh]
    if len(high_missing_rate) > 0:
        logging.info(f'Columns with more than {high_missing_thresh:.0%} missing values: {high_missing_rate.index.tolist()}')
    else:
        logging.info(f'No columns with more than {high_missing_thresh:.0%} of values missing.')


    # produce univariate plots
    logging.info('Producing histograms of numerical variables')
    for i in quant_columns:
        plt.hist(x=df[i])
        plt.title(i)
        plt.savefig(f'images/eda/{i}_histogram.png', bbox_inches="tight")
        plt.clf()
    logging.info(f'Histograms produced and saved in images/eda/ folder')
    
    logging.info('Producing frequency plots of categorical variables')
    for i in cat_cols:
        freq_series = df[i].value_counts(normalize=True)
        plt.bar(x=freq_series.index, 
                height=freq_series)
        plt.title(i)
        plt.xticks(rotation=30)
        plt.savefig(f'images/eda/{i}_frequency_plot.png', bbox_inches="tight")
        plt.clf()
    logging.info(f'Frequeny plots produced and saved in images/eda/ folder')

    
    # bivariate plots
    logging.info('Map continuous variables to target variable')
    for i in quant_columns:
        box_plot = sns.boxplot(data=df, x=target_col, y=i)
        fig = box_plot.get_figure()
        plt.title(i)
        fig.savefig(f'images/eda/{i}_boxplot.png', bbox_inches="tight")
        plt.clf()
    logging.info(f'Boxplots produced and saved in images/eda/ folder')


    logging.info('Map categorical variables to target variable')
    for i in cat_columns:
        plot_series = df.groupby(i)[target_col].mean()
        plt.bar(x=plot_series.index, height=plot_series)
        plt.title(i)
        plt.xlabel(i)
        plt.ylabel(f'Mean {target_col}')
        fig.savefig(f'images/eda/{i}_response_plot.png', bbox_inches="tight")
        plt.clf()
    logging.info(f'Target variable response plots produced and saved in images/eda/ folder')

    logging.info('Produce correlation heatmap')
    sns.heatmap(df[num_cols + ['Churn']].corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig("images/eda/Correlation_heatmap.png", bbox_inches="tight")
    logging.info(f'Correlation heatmap produced and saved in images/eda/ folder')




    
    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    # import data
    df = import_data('data/bank_data.csv')

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
        ]

    # define churn variable
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    logging.info('Churn variable created')

    # eda
    perform_eda(df, num_cols=quant_columns, cat_cols=cat_columns, target_col='Churn')