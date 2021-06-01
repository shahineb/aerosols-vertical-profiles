# Reconstructing Aerosols Vertical Profiles with Aggregate Output Learning

[Download](https://www.dropbox.com/s/4xtunj7h4ufxz7y/final_dataset_lat_321_lon_321_t_6_01_2007.nc?dl=0) data and place under `AODisaggregation/data/` directory.

### Run Aggregate Ridge Regression

```bash
$ python run_ridge_regression.py --cfg=config/ridge_regression.yaml --o=my/output/dir --plot
```

### Run Two Stage Aggregate Ridge Regression

```bash
$ python run_two_stage_ridge_regression.py --cfg=config/two_staged_ridge_regression.yaml --o=my/output/dir --plot
```


## Installation

Code implemented in Python 3.8.0

#### Setting up environment

Create and activate environment
```bash
$ pyenv virtualenv 3.8.0 venv
$ pyenv activate venv
$ (venv)
```

Install dependencies
```bash
$ (venv) pip install -r requirements.txt
```
