# Overview

The goal of this competition is to predict which clients are more likely to default on their loans. The evaluation will favor solutions that are stable over time.



https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability

# Set up data

```
kaggle competitions download -c home-credit-credit-risk-model-stability -p data/original_data

```

Then unzip inside folder original_data

Create the required environment by executing following command:
```
//create venv
python -m venv .venv

//activate .venv
source .venv/Scripts/activate

//upgrade pip
python -m pip install --upgrade pip

//instal package in editable mode
python -m pip install -e .

//clean egg-info artifact
python setup.py clean
```

or simply execute install_all.bat

# How To

TODO