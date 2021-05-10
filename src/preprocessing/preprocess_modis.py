import sys

import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xarray as xr
import dask
import netCDF4 as nc
from pprint import pprint
import pandas as pd
from functools import partial
import multiprocessing as mp


def standardise_coords_and_dims_modis(ds):
    ds = ds.rename({'Cell_Along_Swath:mod04': 'x', 'Cell_Across_Swath:mod04': 'y',
                   'Latitude': 'lat', 'Longitude': 'lon', 'Scan_Start_Time': 'time'})
    ds = ds.set_coords(['time', 'lon', 'lat'])
    return ds


def preprocess_aod_data(ds, 
                        aerosol_qa_cutoff = 1.0, 
                        sza_cutoff = None,
                        verbose=False):
    
    """Takes modis level 2 dataset, filters using AOD 550 QA flag 

    ### TO DO: 
        - check if low SZA should also be filtered        

    ## NOTE: the problem with using cis here was that some data would be flattened without warning.

    Args:
        ds (xarray.Dataset): dataset from one modis l2 granule
        aerosol_qa_cutoff (float): lower limit for Land_Ocean_Quality_Flag
        sza_cutoff(float): upper limit for solar zenith angle

    Returns:
        ds (xr.Dataset)

    """
    
    land_mask = (ds['Land_sea_Flag'] == 0)
    
    # QA flags applied as recommended (Hsu et al., 2013; Levy et al., 2013; Sayer et al. 2013)
    # if Dark Target algo used accept all points w flag > 2
    aod_dt_qa_mask = (ds['AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag'] == 0.) & (ds['AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'] > 2.)
    # if Deep Blue algo used accept all points w flag > 1
    aod_db_qa_mask = (ds['AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag'] == 1.) & (ds['AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'] > 1.)
    aod_dtdb_qa_mask = aod_dt_qa_mask | aod_db_qa_mask
    aod_dtdb_mask = land_mask & aod_dtdb_qa_mask
    
    # this product contains  only  AOD  values  for  the  filtered,
    # quantitatively  useful  retrievals  over  dark  targets
    aod_lo_qa_mask = (ds['Land_Ocean_Quality_Flag'] >= aerosol_qa_cutoff)
    aod_lo_mask = land_mask & aod_lo_qa_mask
    
    if sza_cutoff:
        sza_mask = (ds['Solar_Zenith'] <= sza_cutoff)
        aod_dtdb_mask = aod_dtdb_mask & sza_mask
        aod_lo_mask = aod_lo_mask & sza_mask
    
    # ignore data from this file if no valid points
    valid_points = np.count_nonzero(aod_lo_mask)
    valid_points_dtdb = np.count_nonzero(aod_dtdb_mask)
    
    if verbose:
            print(f'The number of valid points in this dataset from AOD_Land_Ocean is {valid_points} and from DT_DB is {valid_points_dtdb}')

    if min(valid_points, valid_points_dtdb) == 0:
        return None
    
    ds['AOD_550_Dark_Target_Deep_Blue_Combined'] = ds['AOD_550_Dark_Target_Deep_Blue_Combined'].where(aod_dtdb_mask)
    ds['Optical_Depth_Land_And_Ocean'] = ds['Optical_Depth_Land_And_Ocean'].where(aod_lo_mask)
    
    ds = standardise_coords_and_dims_modis(ds)
    ds.attrs['description'] = f"""MODIS AOD 550 from Optical Depth Land Ocean over the oceans (Land_sea_flag = 0),
                                    filtered using QA flag (see  https://doi.org/10.1002/2014JD022453)
                                    with lowest accepted value for Land_Ocean_Quality_Flag {aerosol_qa_cutoff}"""

    return ds[['AOD_550_Dark_Target_Deep_Blue_Combined', 'AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag',
              'Optical_Depth_Land_And_Ocean', 'PSML003_Ocean', 'Solar_Zenith']]



def read_and_process_hdfs(date, 
                          base_dir, 
                          dim, 
                          drop_variables=None, 
                          preprocess_func=None,
                          verbose=False):
    """Creates processed daily dataset by applying the transform function on datasets from
        each satellite swath and concatenating along the specified dimension.
        
        Args:
            date (str or timestamp): date of target hdf files 
            base_dir (str): base directory 
            dim (str): dimension to concatenate the datasets along
            drop_variables (list): list of variables to drop when opening dataset
            preprocess_fund (func): function used to preprocess each dataset before concat
        
        Returns:
            combined (xr.Dataset)
        
        """
    def process_one_path(path):
        
        open_kwargs = dict(decode_cf=True, decode_times=False,
                           drop_variables=drop_variables, mask_and_scale=True)
        
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, **open_kwargs) as ds:
            # transform_func should do some sort of selection or aggregation
            if preprocess_func is not None:
                ds = preprocess_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            if ds is not None: 
                ds.load()
                return ds
            else: 
                return None
    
    date = str(date).split(' ')[0].split('-')
    year = date[0]
    month = date[1]
    day = date[2]
    paths = sorted(glob.glob(base_dir + year + '/' + month + '/' + day + '/' + '*.hdf'))
    
    if verbose:
        print(f'Processing MODIS files for {day}/{month}/{year}')
    datasets = [process_one_path(p) for p in paths]
    datasets = [ds for ds in datasets if ds]
    
    if len(datasets):

        combined = xr.concat(datasets, dim)
        combined.time.attrs['units'] = 'seconds since 1993-01-01'
        combined = xr.decode_cf(combined)
        combined.to_netcdf('/gws/nopw/j04/eo_shared_data_vol2/scratch/sofija/aodisaggregation/data/interim/' +
                           year + '/' + month + '/' + "Processed_MOD_AOD_MYD04_L2_" + day + '_' + month + '_' 
                           + year + ".nc")
        return combined
    
    else:
        print(f'No valid data for {day}/{month}/{year}!')
    

        
        