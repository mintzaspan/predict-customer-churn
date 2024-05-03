# Predict Customer Churn

## Description
Identify credit card customers that are most likely to churn. This project uses data on customer demographics, economic status and financial history to predict attrition.

## How to use this repo

### Software requirements 
- Git
- Conda

### Execution steps
```bash
# clone the repo
git clone https://github.com/mintzaspan/predict-customer-churn.git

# change to project directory
cd predict-customer-churn

# install conda environment
conda env create -f environment.yml

# activate conda environment
conda activate predict-customer-churn

# export current directory to python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# set up for pre-commit hooks
pre-commit install

# run tests
pytest

# run churn module
python src/churn_library.py
```

### Structure
After executing the commands above the repo folder will be populated as shown below. 
```
predict-customer-churn/
│   README.md
│   pytest.ini
│   LICENSE  
│   .gitignore
│   .environment.yml
│   .pre-commit-config.yaml
│
├── data/
│   │   bank_data.csv
│
├── src/
│   │   churn_library.py
│   │   config.yaml
│
├── tests/
│   │   test_churn_script_logging_and_tests.py
│   │   conftest.py
│   ├── logs/
│       │   <churn_library>.log
│
├── models/
│   │   *.pkl
│
├── images/
│   ├── eda/
│       │   <univariate_plots>.png
│       │   <bivariate_plots>.png
│   ├── results/
│       │   <classification_reports>.png
│       │   <ROC_AUC_comparison_graph>.png
│       │   <feature_importance_graphs>.png
└──
```

