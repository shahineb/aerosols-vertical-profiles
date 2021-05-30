# Path to configuration file
CFG_PATH=config/ridge_regression.yaml

# Define output directories path variables
OUTPUT_DIR=data/experiment_outputs/ridge_regression

# Run experiment
python run_ridge_regression.py --cfg=$CFG_PATH --o=$OUTPUT_DIR
