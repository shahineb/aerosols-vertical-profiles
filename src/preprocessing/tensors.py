import torch

standardize = lambda x: (x - x.mean()) / x.std()


def make_grid_tensor(field, coords_keys, standardize=True):
    """Makes ND tensor grid corresponding to
    provided xarray dataarray coordinates
    """
    coords = []
    for key in coords_keys:
        coord = torch.from_numpy(field[key].values.astype('float'))
        if standardize:
            coord = standardize(coord)
        coords.append(coord)
    grid = torch.stack(torch.meshgrid(*coords), dim=-1).float()
    return grid
