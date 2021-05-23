"""
Description : Runs Variational Warped Conditional Integral Process regression experiment

Usage: run_logwarped_ridge_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Index of GPU to use [default: 0].
  --lr=<lr>                        Learning rate.
  --n_epochs=<n_epochs>            Number of training epochs.
  --log_every=<frequency>          Dump logs every X epoch [default: 1].
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
from src.models import VariationalCIP
from src.likelihoods import CMPLikelihood
from src.mlls import AggregateVariationalELBO
from src.evaluation import metrics, visualization


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
    h_grid = preproc.make_3d_groundtruth_tensor(dataset=dataset, groundtruth_variable_key='h')
    x_grid = preproc.make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'])
    y_grid = preproc.make_2d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['2d_covariates']))
    z_grid = preproc.make_2d_target_tensor(dataset=standard_dataset, target_variable_key=cfg['dataset']['target'])
    gt_grid = preproc.make_3d_groundtruth_tensor(dataset=dataset, groundtruth_variable_key=cfg['dataset']['groundtruth'])

    # Reshape tensors
    h = h_grid.reshape(-1, x_grid.size(-2))
    x_by_bag = x_grid.reshape(-1, x_grid.size(-2), x_grid.size(-1))
    x = x_by_bag.reshape(-1, x_grid.size(-1))
    y = y_grid.reshape(-1, y_grid.size(-1))
    z = z_grid.flatten()
    gt = gt_grid.flatten()
    return dataset, standard_dataset, h, h_grid, x_by_bag, x, y_grid, y, z_grid, z, gt_grid, gt


def make_model(cfg, x):
    # Initialize inducing points regularly across samples
    n_samples = x.size(0)
    step = n_samples // cfg['model']['n_inducing_points']
    offset = (n_samples % cfg['model']['n_inducing_points']) // 2
    inducing_points = x[offset:n_samples - offset:step].float()

    # Initialize mean module
    mean_module = gpytorch.means.ZeroMean()

    # Initialize kernel on x
    kernel_x = gpytorch.kernels.LinearKernel()

    # Initialize kernel on y
    kernel_y = gpytorch.kernels.LinearKernel()

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

    # Define model
    model = VariationalCIP(inducing_points=inducing_points,
                           individuals_mean=mean_module,
                           individuals_kernel=kernel_x,
                           bag_kernel=kernel_y,
                           transform=transform,
                           lbda=cfg['model']['lbda'])
    return model


def fit(cfg, model, h, x_by_bag, y, z, h_grid, groundtruth_3d, targets_2d, log_every, plot, plot_every, output_dir, device_idx):
    # Transfer on device
    device = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
    h = h.to(device)
    x_by_bag = x_by_bag.to(device)
    y = y.to(device)
    z = z.to(device)

    # Define an infinite sampler of 3d covariates columns with column 2d covariates
    def sampler_xy(batch_size):
        buffer = torch.ones(x_by_bag.size(0))
        while True:
            idx = buffer.multinomial(batch_size)
            x1 = x_by_bag[idx].reshape(-1, x_by_bag.size(-1))
            y1 = y[idx]
            h1 = h[idx]
            yield x1, y1, h1

    # Define an iterator over columns 2d covarites and aggregate 2d targets
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(n_bags)
        sampler_1 = sampler_xy(batch_size=cfg['training']['batch_size_xy'])
        for idx in rdm_indices.split(batch_size):
            x1, y1, h1 = next(sampler_1)
            y2 = y[idx]
            z2 = z[idx]
            yield x1, y1, h1, y2, z2

    # Define likelihood module
    likelihood = CMPLikelihood()

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=cfg['training']['lr'])

    # Loss
    elbo = AggregateVariationalELBO(likelihood, model, num_data=len(z), beta=cfg['model']['beta'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    batch_size = cfg['training']['batch_size']
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()

    # Metrics record
    logs = dict()

    for epoch in range(cfg['training']['n_epochs']):

        batch_bar = Bar("Batch", max=len(z) // batch_size)
        epoch_loss = 0

        for i, (x1, y1, h1, y2, z2) in enumerate(batch_iterator(batch_size=batch_size)):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(x1)

            # Compute negative ELBO loss
            buffer = model.get_elbo_computation_parameters(bags_covariates_1=y1, bags_covariates_2=y2)
            loss = -elbo(variational_dist_f=q,
                         target=z2,
                         transform=model.transform,
                         aggregation_support=h1,
                         **buffer)

            # Take gradient step
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            batch_bar.suffix = f"Running ELBO {-epoch_loss / (i + 1)}"
            batch_bar.next()

        # Whether logs and plots must me dumped at this epoch
        dump_epoch_logs = epoch % log_every == 0
        dump_epoch_plots = plot and epoch % plot_every == 0

        if dump_epoch_logs or dump_epoch_plots:
            # Run inference on all 3D covariates
            prediction_3d = predict(model, x)

            # Compute metrics and dump logs if needed
            if dump_epoch_logs:
                epoch_logs = get_epoch_logs(model, likelihood, prediction_3d, groundtruth_3d, targets_2d, h_grid)
                epoch_logs.update({'loss': epoch_loss / (len(z) // batch_size)})
                with open(os.path.join(output_dir, 'running_logs.yaml'), 'w') as f:
                    yaml.dump({'epoch': logs}, f)

            # Dump plots if needed
            if dump_epoch_plots:
                dump_plots(cfg=cfg, dataset=dataset, prediction_3d=prediction_3d, aggregation_support=h_grid, output_dir=os.path.join(output_dir, f'png/epoch_{epoch}'))

        # Complete progress bar
        epoch_bar.next()
        epoch_bar.finish()

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(output_dir, 'state.pt'))



def get_epoch_logs(model, likelihood, prediction_3d, groundtruth_3d, targets_2d, h):
    # Define agggregation function
    def aggregate_fn(grid):
        int_grid = -torch.trapz(y=grid, x=h, dim=-1)
        return int_grid

    scores = metrics.compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn)
    return scores


def dump_plots(cfg, dataset, prediction_3d, aggregation_support, output_dir):
    # Define agggregation function
    def aggregate_fn(grid):
        int_grid = -torch.trapz(y=grid, x=h, dim=-1)
        return int_grid

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



def update_cfg(cfg, args):
    """Updates loaded configuration file with specified command line arguments

    Args:
        cfg (dict): loaded configuration file
        args (dict): script execution arguments

    Returns:
        type: dict

    """
    if args['--lr']:
        cfg['training']['lr'] = float(args['--lr'])
    if args['--n_epochs']:
        cfg['training']['n_epochs'] = int(args['--n_epochs'])
    return cfg



if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_cfg(cfg, args)

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