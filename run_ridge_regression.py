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
from docopt import docopt
import torch
import matplotlib.pyplot as plt
import src.preprocessing as preproc
from src.models import AggregateRidgeRegression
from src.evaluation import metrics, visualization
import numpy as np


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    dataset, standard_dataset, x_by_bag, x, z_grid, z, gt_grid, gt, h_grid, h = make_datasets(cfg=cfg)

    # Instantiate model
    model = make_model(cfg=cfg, dataset=standard_dataset, h=h)
    logging.info(f"{model}")

    # Fit model
    model.fit(x_by_bag, z)
    logging.info("Fitted model")

    h_std = dataset.h.std().values
    target_mean = torch.tensor(dataset[cfg['dataset']['target']].mean().values)
    target_std = torch.tensor(dataset[cfg['dataset']['target']].std().values)
    print(f'h standard dev is {h_std}')
    # Run prediction
    with torch.no_grad():
        prediction = model(x)
        prediction_3d = prediction.reshape(*gt_grid.shape)
        
        # destandardise 3d prediction
        prediction_3d_dest = target_std * (prediction_3d + target_mean) / h_std

    # Dump scores in output dir
    dump_scores(
                prediction_3d=prediction_3d,
                groundtruth_3d=gt_grid,
                targets_2d=z_grid,
                aggregate_fn=model.aggregate_fn,
                output_dir=args['--o'])

    # Dump plots in output dir
    if args['--plot']: 
        dump_plots(cfg=cfg,
                   dataset=dataset,
                   standard_dataset=standard_dataset, 
                   prediction_3d=prediction_3d,
                   prediction_3d_dest=prediction_3d_dest,
                   aggregate_fn=model.aggregate_fn,
                   output_dir=args['--o'])
        logging.info("Dumped plots")


def make_datasets(cfg):
    # Load dataset
    dataset = preproc.load_dataset(file_path=cfg['dataset']['path'],
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
    gt_grid = preproc.make_3d_groundtruth_tensor(dataset=dataset, groundtruth_variable_key=cfg['dataset']['groundtruth'])
    h_grid = preproc.make_3d_groundtruth_tensor(dataset=standard_dataset, groundtruth_variable_key='h')

    # Reshape tensors
    x_by_bag = x_grid.reshape(-1, x_grid.size(-2), x_grid.size(-1))
    x = x_by_bag.reshape(-1, x_grid.size(-1))
    z = z_grid.flatten()
    gt = gt_grid.flatten()
    h = h_grid.reshape(-1, x_grid.size(-2))
    

    return dataset, standard_dataset, x_by_bag, x, z_grid, z, gt_grid, gt, h_grid, h

def make_model(cfg, dataset, h):

    def trpz(grid):
        # Create aggregation operator
#         h = preproc.standardize(dataset.h[0, :, 0, 0].values)
#         std_h_grid = torch.from_numpy(h)
#         h_grid = torch.from_numpy(preproc.standardize(h)).float()

        int_grid = -torch.trapz(y=grid, x=h.unsqueeze(-1), dim=-2)
        return int_grid

    # Instantiate model
    model = AggregateRidgeRegression(alpha=cfg['model']['alpha'],
                                     aggregate_fn=trpz,
                                     fit_intercept=cfg['model']['fit_intercept'])
    return model

# def predict(model, x, dataset, cfg, gt_grid):
#     h_std = torch.tensor(dataset.h.std().values
#     target_mean = torch.tensor(dataset[cfg['dataset']['target']].mean().values)
#     target_std = torch.tensor(dataset[cfg['dataset']['target']].std().values)

#     with torch.no_grad():
#         prediction = model(x)
#         print(f'pred max before destandard is {torch.max(prediction)}')
#         prediction_3d = prediction.reshape(*gt_grid.shape)
#         prediction_3d_dest = prediction_3d / h_std
#         print(f'pred max after destandard is {torch.max(prediction_3d_dest)}')
#         print(f'max of ground truth is {torch.max(gt)}')



def dump_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn, output_dir):
    scores = metrics.compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn)
    dump_path = os.path.join(output_dir, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"Dumped scores at {dump_path}")


def dump_plots(cfg, dataset, standard_dataset, prediction_3d, prediction_3d_dest, aggregate_fn, output_dir):
    # First plot - aggregate 2D prediction
    dump_path = os.path.join(output_dir, 'aggregate_2d_prediction.png')
    _ = visualization.plot_aggregate_2d_predictions(dataset=standard_dataset,
                                                    target_key=cfg['dataset']['target'],
                                                    prediction_3d=prediction_3d,
                                                    aggregate_fn=aggregate_fn)
    plt.savefig(dump_path)
    plt.close()

    # Second plot - slices of covariates
    dump_path = os.path.join(output_dir, 'covariates_slices.png')
    _ = visualization.plot_vertical_covariates_slices(dataset=dataset,
                                                      lat_idx=cfg['evaluation']['slice_latitude_idx'],
                                                      time_idx=cfg['evaluation']['slice_time_idx'],
                                                      covariates_keys=cfg['evaluation']['slices_covariates'])
    plt.savefig(dump_path)
    plt.close()

    # Third plot - prediction slice
    dump_path = os.path.join(output_dir, '3d_prediction_slice.png')
    _ = visualization.plot_vertical_prediction_slice(dataset=dataset,
                                                     lat_idx=cfg['evaluation']['slice_latitude_idx'],
                                                     time_idx=cfg['evaluation']['slice_time_idx'],
                                                     groundtruth_key=cfg['dataset']['groundtruth'],
                                                     prediction_3d=prediction_3d_dest)
    plt.savefig(dump_path)
    plt.close()


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
