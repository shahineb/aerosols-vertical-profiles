import numpy as np
import xarray as xr

standardize = lambda x: (x - x.mean()) / x.std()

def load_dataset(file_path, trimming_altitude_idx):
    # Load dataset
    dataset = xr.open_dataset(file_path).isel(lat=slice(200, 350),
                                              lon=slice(200, 350),
                                              time=slice(0, 3))

    # Set h as altitude coordinate
#     dataset = dataset.assign_coords(h=('lev', dataset.h.isel(time=0, lat=0, lon=0).values))

    # Trim altitude
    dataset = dataset.isel(lev=slice(trimming_altitude_idx, len(dataset.lev)))

    return dataset


def make_groundtruh_field(dataset):
    # To be updated if we change fields
    dataset['so4_mass_conc'] = dataset.so4 * dataset.airdens
    return dataset


def to_log_domain(dataset, variables_keys):
    for key in variables_keys:
        dataset['log_' + key] = np.log(dataset[key])
    return dataset
