"""
Description : Runs ridge regression experiment

Usage: run_ridge_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs scatter plots.
"""
import os
import yaml
import logging
import docopt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import src.preprocessing as preproc
from src.models import AggregateRidgeRegression
from src.evaluation import metrics, visualization


def make_datasets(cfg):
    # Load dataset
    dataset = preproc.load_dataset(file_path=cfg['dataset']['path'],
                                   air_density_file_path=cfg['dataset']['air_density_file_path'],
                                   trimming_altitude_idx=cfg['dataset']['trimming_altitude_idx'])

    # Compute groundtruth 3D+t field
    dataset = preproc.make_groundtruh_field(dataset=dataset)

    # Apply logtransform to specified variables
    dataset = preproc.to_log_domain(dataset=dataset, variables_keys=cfg['dataset']['to_log_domain'])

    # Compute standardized versions
    standard_dataset = preproc.standardize(dataset)

    # Convert into pytorch tensors
    x_grid = preproc.make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'])
    z_grid = preproc.make_2d_target_tensor(dataset=standard_dataset, target_variable_key=cfg['dataset']['target'])
    gt_grid = preproc.make_3d_groundtruth_tensor(dataset=standard_dataset, groundtruth_variable_key=cfg['dataset']['groundtruth'])

    # Reshape tensors
    x_by_bag = x_grid.reshape(-1, x_grid.size(-2), x_grid.size(-1))
    x = x_by_bag.reshape(-1, x_grid.size(-1))
    z = z_grid.flatten()
    gt = gt_grid.flatten()

    return dataset, standard_dataset, x_by_bag, x, z_grid, z, gt_grid, gt




if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
