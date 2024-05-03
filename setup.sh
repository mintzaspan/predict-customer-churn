# install conda environment
conda env create -f environment.yml

# activate conda environment
conda activate predict-customer-churn

# export pwd to python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# run tests
pytest

# run churn module
python scripts/churn_library.py