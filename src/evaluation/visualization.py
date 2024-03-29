import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_2d_covariates(dataset, time_idx, covariates_keys):
    """Plots next to each other lat/lon fields of 2D covariates

    Args:
        dataset (xr.Dataset): source dataset
        time_idx (int): index of time to use for slice
        covariates_keys (list[str]): list of variable names to plot

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """

    field_set = dataset.isel(time=time_idx)
    lon = dataset.lon.values
    lat = dataset.lat.values
    
    nrows = len(covariates_keys)
    fig, ax = plt.subplots(nrows, 1, figsize=(5 * nrows, 5 * nrows))
    cmap = 'magma'
    n_x_ticks = 100
    n_y_ticks = 100
    title_fontsize = 20
    labels_fontsize = 12
    cbar_fontsize = 12

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(field_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), n_x_ticks))
        ax[i].set_xticklabels(lon[::n_x_ticks], Fontsize=cbar_fontsize)
        ax[i].set_yticks(range(0, len(lat), n_y_ticks))
        ax[i].set_yticklabels(lat[::n_x_ticks], rotation=10, Fontsize=cbar_fontsize)
        ax[i].set_title(key, fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="vertical", ax=ax[i], shrink=0.8)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    ax[0].set_xlabel('longitude', fontsize=labels_fontsize)
    ax[0].set_ylabel('latitude', fontsize=labels_fontsize)
    
    plt.tight_layout()
    return fig, ax

def plot_aggregate_2d_predictions(dataset, target_key, prediction_3d, aggregate_fn):
    """Plots aggregation of 3D+t prediction, 2D+t aggregate targets used for training and difference

    Args:
        dataset (xr.Dataset): source dataset
        target_key (str): name of target variable
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lon)

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """
    n_row = prediction_3d.size(0)
    n_col = prediction_3d.size(0)*prediction_3d.size(1)*prediction_3d.size(2)
    prediction_3d_grid = prediction_3d.reshape(n_col, -1)
    
    aggregate_prediction_2d = aggregate_fn(prediction_3d_grid.unsqueeze(-1)).squeeze()
    aggregate_prediction_2d = aggregate_prediction_2d.reshape(n_row, prediction_3d.size(1), prediction_3d.size(2))

    fig, ax = plt.subplots(n_row, 3, figsize=(5 * n_row, 5 * n_row))
    cmap = 'magma'
    title_fontsize = 16
    cbar_fontsize = 12

    for i in range(n_row):
        groundtruth = dataset.isel(time=i)[target_key].values
        pred = aggregate_prediction_2d[i].numpy()
        difference = groundtruth - pred
        vmin = np.minimum(groundtruth, pred).min()
        vmax = np.maximum(groundtruth, pred).max()
        diffmax = np.abs(difference).max()
        rmse = round(np.sqrt(np.mean(difference ** 2)), 4)

        im = ax[i, 0].imshow(groundtruth, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, orientation="horizontal", ax=ax[i, 0], shrink=0.5)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f'Groundtruth - Time step {i}', fontsize=title_fontsize)

        im = ax[i, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i, 1].axis('off')
        ax[i, 1].set_title(f'Prediction - Time step {i}', fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="horizontal", ax=ax[i, 1], shrink=0.5)
        cbar.ax.tick_params(labelsize=cbar_fontsize)

        im = ax[i, 2].imshow(difference, cmap='bwr', vmin=-diffmax, vmax=diffmax)
        ax[i, 2].axis('off')
        ax[i, 2].set_title(f'Difference - RMSE {rmse}', fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="horizontal", ax=ax[i, 2], shrink=0.5)
        cbar.ax.tick_params(labelsize=cbar_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_vertical_covariates_slices(dataset, lat_idx, time_idx, covariates_keys):
    """Plots next to each other lon/alt slices of 3D covariates

    Args:
        dataset (xr.Dataset): source dataset
        lat_idx (int): index of latitude to use for slice
        time_idx (int): index of time to use for slice
        covariates_keys (list[str]): list of variable names to plot

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """
    slice_set = dataset.isel(lat=lat_idx, time=time_idx)
    h = slice_set.isel(lon=10).h.values
    lon = dataset.lon.values

    nrows = len(covariates_keys)
    fig, ax = plt.subplots(nrows, 1, figsize=(2 * nrows, 4 * nrows))
    cmap = 'magma'
    n_x_ticks = 20
    n_y_ticks = 20
    title_fontsize = 26
    labels_fontsize = 18
    cbar_fontsize = 18

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(slice_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), n_x_ticks))
        ax[i].set_xticklabels(lon[::n_x_ticks])
        ax[i].set_yticks(range(0, len(h), n_y_ticks))
        ax[i].set_yticklabels(h[::n_x_ticks], rotation=10)
        ax[i].set_title(key, fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="vertical", ax=ax[i], shrink=0.7)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    ax[0].set_xlabel('longitude', fontsize=labels_fontsize)
    ax[0].set_ylabel('altitude', fontsize=labels_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_vertical_prediction_slice(dataset, lat_idx, time_idx, groundtruth_key, prediction_3d):
    """Plots lon/alt slice of 3D prediction next to groundtruth, difference and RMSE as a function of height

    Args:
        dataset (xr.Dataset): source dataset
        lat_idx (int): index of latitude to use for slice
        time_idx (int): index of time to use for slice
        groundtruth_key (str): name of groundtruth variable
        prediction_3d (torch.Tensor): (time, lat, lon, lev)

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """
    
    h = dataset.isel(lat=lat_idx, time=time_idx, lon=0).h.values
    lon = dataset.lon.values

    predicted_slice = prediction_3d[time_idx, lat_idx]
    groundtruth_slice = dataset.isel(lat=lat_idx, time=time_idx)[groundtruth_key].values.T

    difference = groundtruth_slice - predicted_slice.numpy()
    squared_error = difference ** 2
    total_rmse = round(np.sqrt(np.mean(squared_error)), 4)

    vmin = min(groundtruth_slice.min(), predicted_slice.min())
    vmax = max(groundtruth_slice.max(), predicted_slice.max())
    diffmax = np.abs(difference).max()

    fig, ax = plt.subplots(4, 1, figsize=(10, 18))
    cmap = 'magma'
    n_x_ticks = 20
    n_y_ticks = 20
    title_size = 22
    label_size = 18

    im = ax[0].imshow(groundtruth_slice.T, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, orientation="horizontal", ax=ax[0], shrink=0.7)
    cbar.ax.tick_params(labelsize=label_size)
    ax[0].set_title(f'Groundtruth \n ({groundtruth_key})', fontsize=title_size)

    im = ax[1].imshow(predicted_slice.T, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, orientation="horizontal", ax=ax[1], shrink=0.7)
    cbar.ax.tick_params(labelsize=label_size)
    ax[1].set_xticks(range(0, len(lon), 20))
    ax[1].set_xticklabels(lon[::20])
    ax[1].set_yticks(range(0, len(h), 20))
    ax[1].set_yticklabels(h[::20], rotation=10)
    ax[1].set_xlabel('longitude', fontsize=label_size)
    ax[1].set_ylabel('altitude', fontsize=label_size)
    ax[1].set_title('Prediction', fontsize=title_size)

    im = ax[2].imshow(difference.T, cmap='bwr', vmin=-diffmax, vmax=diffmax)
    cbar = plt.colorbar(im, orientation="horizontal", ax=ax[2], shrink=0.7)
    cbar.ax.tick_params(labelsize=label_size)
    ax[2].set_xticks(range(0, len(lon), 20))
    ax[2].set_xticklabels(lon[::20])
    ax[2].set_yticks(range(0, len(h), 20))
    ax[2].set_yticklabels(h[::20], rotation=10)
    ax[2].set_xlabel('longitude', fontsize=label_size)
    ax[2].set_ylabel('altitude', fontsize=label_size)
    ax[2].set_title('Difference', fontsize=title_size)

    for i in range(3):
        ax[i].set_xticks(range(0, len(lon), n_x_ticks))
        ax[i].set_xticklabels(lon[::n_x_ticks])
        ax[i].set_yticks(range(0, len(h), n_y_ticks))
        ax[i].set_yticklabels(h[::n_y_ticks], rotation=10)
        ax[i].set_xlabel('longitude', fontsize=label_size)
        ax[i].set_ylabel('altitude', fontsize=label_size)

    ax[3].plot(h, np.sqrt(np.mean(squared_error, axis=0)), '--.', label=f'RMSE={total_rmse}')
    ax[3].grid(alpha=0.5)
    ax[3].set_xlabel('altitude', fontsize=label_size)
    ax[3].set_ylabel('RMSE', fontsize=label_size)
    ax[3].set_title("RMSE profile", fontsize=title_size)

    plt.legend(fontsize=label_size)
    plt.tight_layout()
    return fig, ax