# Path to configuration file
CFG_PATH=config/warped_ridge_regression.yaml

# Define output directories path variables
OUTPUT_DIR=data/experiment_outputs/warped_ridge_regression

# Run experiments for multiple seeds
for SEED in 2 3 5 7 9 11 13 17 19 23 29 31 37 41 43 47 53 59 73 79;
do
  DIRNAME=seed_$SEED
  python run_warped_ridge_regression.py --cfg=$CFG_PATH --o=$OUTPUT_DIR/$DIRNAME --seed=$SEED
done
