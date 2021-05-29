"""
Description : Runs warped ridge regression experiment

Usage: run_warped_ridge_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs scatter plots.
"""
import os
import yaml
import logging
from docopt import docopt
from progress.bar import Bar
import torch
import matplotlib.pyplot as plt
import src.preprocessing as preproc
from src.models import TransformedAggregateRidgeRegression
from src.evaluation import metrics, visualization


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    dataset, standard_dataset, x_by_bag, x, z_grid, z_grid_std, z, gt_grid, gt, h_grid, h = make_datasets(cfg=cfg)
   
    # Instantiate model
    model = make_model(cfg=cfg, dataset=dataset, h=h)
    logging.info(f"{model}")

    # Fit model
    model = fit(cfg=cfg, model=model, x=x, x_by_bag=x_by_bag, z=z, z_grid=z_grid)
    logging.info("Fitted model")

    # Run prediction
    with torch.no_grad():
        prediction = model(x)
        prediction_3d = prediction.reshape(*gt_grid.shape)

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d,
                groundtruth_3d=gt_grid,
                targets_2d=z_grid,
                aggregate_fn=model.aggregate_fn,
                output_dir=args['--o'])

    # Dump plots in output dir
    if args['--plot']:
        dump_plots(cfg=cfg,
                   dataset=dataset,
                   prediction_3d=prediction_3d,
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
    z_grid_std = preproc.make_2d_target_tensor(dataset=standard_dataset, target_variable_key=cfg['dataset']['target'])
    z_grid = preproc.make_2d_target_tensor(dataset=dataset, target_variable_key=cfg['dataset']['target'])
    gt_grid = preproc.make_3d_groundtruth_tensor(dataset=dataset, groundtruth_variable_key=cfg['dataset']['groundtruth'])
    h_grid = preproc.make_3d_groundtruth_tensor(dataset=dataset, groundtruth_variable_key='h')

    # Reshape tensors
    x_by_bag = x_grid.reshape(-1, x_grid.size(-2), x_grid.size(-1))
    x = x_by_bag.reshape(-1, x_grid.size(-1))
    z = z_grid_std.flatten()
    gt = gt_grid.flatten()
    h = h_grid.reshape(-1, x_grid.size(-2))
    
    return dataset, standard_dataset, x_by_bag, x, z_grid, z_grid_std, z, gt_grid, gt, h_grid, h


def make_model(cfg, dataset, h): 
    target_variable_key = cfg['dataset']['target']
    target_mean, target_std = torch.tensor(dataset[target_variable_key].mean().values), torch.tensor(dataset[target_variable_key].std().values)
    
    # Define an aggregation operator over the entire grid used for evaluation
    def trpz(grid):        
        int_grid = -torch.trapz(y=grid, x=h.unsqueeze(-1), dim=-2)
        return int_grid

    # Define warping transformation
    if cfg['model']['transform'] == 'linear':
        transform = lambda x: x
    elif cfg['model']['transform'] == 'softplus':
        transform = lambda x: torch.log(1 + torch.exp(x))
    elif cfg['model']['transform'] == 'smooth_abs':
        transform = lambda x: torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')
    elif cfg['model']['transform'] == 'square':
        transform = torch.square
    elif cfg['model']['transform'] == 'exp':
        transform = torch.exp
    else:
        raise ValueError("Unknown transform")

    # Instantiate model
    model = TransformedAggregateRidgeRegression(alpha=cfg['model']['alpha'],
                                                ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                                aggregate_fn=trpz,
                                                transform=transform,
                                                fit_intercept=cfg['model']['fit_intercept'])
    return model


def fit(cfg, model, x, x_by_bag, z, z_grid):
    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)
    
    z_mean = z_grid.mean()
    z_std = z_grid.std()
    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        prediction = model(x)
        prediction_3d = prediction.reshape(*x_by_bag.shape[:-1])
        aggregate_prediction_2d = model.aggregate_prediction(prediction_3d.unsqueeze(-1)).squeeze()
        aggregate_prediction_2d = (aggregate_prediction_2d - z_mean) / z_std

        # Compute loss
        loss = torch.square(aggregate_prediction_2d - z).mean()
        loss += model.regularization_term()

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item()}"
        bar.next()

    return model


def dump_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn, output_dir, h_std=None):
    scores = metrics.compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn, h_std=h_std)
    dump_path = os.path.join(output_dir, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"Dumped scores at {dump_path}")


def dump_plots(cfg, dataset, prediction_3d, aggregate_fn, output_dir):
    # First plot - aggregate 2D prediction
    dump_path = os.path.join(output_dir, 'aggregate_2d_prediction.png')
    _ = visualization.plot_aggregate_2d_predictions(dataset=dataset,
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
                                                     prediction_3d=prediction_3d)
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
