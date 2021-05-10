import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import dask
import netCDF4 as nc
from pathlib import Path

os.chdir('/home/users/sofija/AODisaggregation/src/preprocessing/')

def _apply_basic_mask(data_array):
    """Masks data that's missing or outside of the valid range.
    
    ##NOTE: xarray mask and scale doesn't work bc CALIOP files don't
        have the correct attribute names 
    
    """
    
    if 'fillvalue' in data_array.attrs:
        fill = data_array.attrs['fillvalue']
        data_array = data_array.where(data_array != fill)
    
    valid_rng = data_array.attrs['valid_range']

    smin, smax = valid_rng.split("...")
    min_valid = float(smin.replace(',', ''))
    max_valid = float(smax.replace(',', ''))
    
    cond = (min_valid <= data_array) & (data_array <= max_valid)
    
    return data_array.where(cond)

def _get_daynight_mask(dn_flag, day_night, alt_dim):
    """Masks nightime (daytime) data if day (night) specified.
    
    DN field indicates the lighting conditions at an
    altitude of ~24 km above mean sea level; 0 = day, 1 = night. 
    
    """
    
    if day_night == 'd':
        dn_cond = np.tile(dn_flag == 0, (1, alt_dim))
    else:
        dn_cond = np.tile(dn_flag == 1, (1, alt_dim))
    
    return dn_cond

def _find_good_qc_points(flag1, flag2, alt_dim):
    """Masks data that's not error-free according to the level 1 QC flags."""
    return (np.tile(flag1 == 0, (1, alt_dim)) & np.tile(flag2 == 0, (1, alt_dim)))

def _find_good_profiles(surface_sat_flag, neg_anomaly_flag, alt_dim):
    """Masks data where the signal is saturated or a negative anomaly detected."""
    return (np.tile(surface_sat_flag == 0, (1, alt_dim)) & np.tile(neg_anomaly_flag == -1, (1, alt_dim)))
    
def _rename_and_drop_fake_dims(data_array):
    """Give CALIOP dimensions sensible names and drop redundant fake dims."""
    
    if data_array.shape[1] == 1:
        return data_array.rename({data_array.dims[0]: 'time'}).squeeze()
    elif data_array.shape[1] == 33:
        return data_array.rename({data_array.dims[0]: 'time', data_array.dims[1]: 'met_alt'})
    elif data_array.shape[1] == 583:
        return data_array.rename({data_array.dims[0]: 'time', data_array.dims[1]: 'alt'})
    else: 
        return data_array.rename({data_array.dims[0]: 'time'})

def _get_land_mask(land_water_flag, alt_dim):
    """Masks data not classified as ocean."""
    
    shallow_ocean = np.tile(land_water_flag == 0, (1, alt_dim))
    cont_ocean = np.tile(land_water_flag == 6, (1, alt_dim))
    deep_ocean = np.tile(land_water_flag == 7, (1, alt_dim))
    
    return (shallow_ocean | cont_ocean | deep_ocean)

def standardise_coords_and_dims_caliop(ds, lidar_altitude, met_altitude):
    
    # fix coordinates and dimensions
    ds = ds.apply(_rename_and_drop_fake_dims)
    ds = ds.set_coords(['Profile_Time', 'Profile_UTC_Time', 'Latitude', 'Longitude'])
    
    # add lidar and meteorological altitude dimensions
    altitude = xr.DataArray(lidar_altitude, dims='alt', attrs={'units': 'km'})
    met_altitude = xr.DataArray(met_altitude, dims='met_alt')
    ds = ds.assign_coords({'alt':('alt', altitude), 'met_alt':('met_alt', met_altitude)})
    ds.alt.attrs['units'] = 'km'
    ds.met_alt.attrs['units'] = 'km'
    
    # make profile time a dim coord and rename using CF convention
    ds = ds.rename({'Profile_Time': 'time'})
    ds.time.attrs['units'] = 'seconds since 1993-01-01'

    return ds

# this is a hack for keeping lidar and meteorology altitudes same across all datasets
# assuiming the errors introduced this way are not that important over the oceans 
alt_post2007 = np.loadtxt(Path('lidar_altitudes/lidar_alt_post2007.asc').resolve())
met_alt = np.loadtxt(Path('lidar_altitudes/met_alt.asc').resolve())

def preprocess_level1_data(ds, 
                           lidar_altitude=alt_post2007, 
                           met_altitude=met_alt, 
                           day_night='d'):
    """Takes calipso level 1 dataset, filters using quality assessment flags, fixes dims and coords
    for lidar and meteorology data and returns a processed dataset.
    
    ### TO DO:
        - consider spacecraft orientation filters (SZA, Scattering angle etc.)
        - check where it makes most sense to decode CF time
    
    Args:
        ds (xarray.Dataset): dataset from one calipso l1 file
        lidar_altitude (np.array): array with altitude data from calipso HDF metadata
        met_altitude (np.array): array with altitude from meteorological data from calipso HDF metadata
        dn_flag ({'d', 'n'}): 
                        'd' keep daytime data
                        'n' keep nighttime data
    Returns:
        ds (xarray.Dataset)
    """
    
    ds = ds.apply(_apply_basic_mask)
    
    alt_dim = len(lidar_altitude)

    mask_dn = _get_daynight_mask(ds['Day_Night_Flag'], day_night, alt_dim)
    
    mask_land = _get_land_mask(ds['Land_Water_Mask'], alt_dim)
    
    mask_qc = _find_good_qc_points(ds["QC_Flag"], ds['QC_Flag_2'], alt_dim)
    
    mask_par = _find_good_profiles(ds['Surface_Saturation_Flag_532Par'],
                                   ds['Negative_Signal_Anomaly_Index_532Par'], alt_dim)
    
    mask_perp = _find_good_profiles(ds['Surface_Saturation_Flag_532Perp'],
                                    ds['Negative_Signal_Anomaly_Index_532Perp'], alt_dim)
    
    mask_1064 = _find_good_profiles(ds['Surface_Saturation_Flag_1064'],
                                    ds['Negative_Signal_Anomaly_Index_1064'], alt_dim)
    
    # total backscatter needs both parallel and perpendicular masks  
    mask_total532_profile = mask_dn & mask_land & mask_qc & mask_par & mask_perp
    mask_perp532_profile = mask_dn & mask_land & mask_qc & mask_perp
    mask_1064_profile = mask_dn & mask_land & mask_qc & mask_1064
    
    # if all values are masked ignore this granule
    if np.count_nonzero(mask_total532_profile) == 0:
        return None
    
    ds['Total_Attenuated_Backscatter_532'] = ds['Total_Attenuated_Backscatter_532'].where(mask_total532_profile)
    ds['Perpendicular_Attenuated_Backscatter_532'] = ds['Perpendicular_Attenuated_Backscatter_532'].where(mask_perp532_profile)
    ds['Attenuated_Backscatter_1064'] = ds['Perpendicular_Attenuated_Backscatter_532'].where(mask_1064_profile)
    
    ds = standardise_coords_and_dims_caliop(ds, lidar_altitude, met_altitude)
    ds = ds[['Total_Attenuated_Backscatter_532', 'Perpendicular_Attenuated_Backscatter_532',
                'Attenuated_Backscatter_1064', 'Temperature', 'Pressure', 'Relative_Humidity',
              'Solar_Zenith_Angle', 'Tropopause_Height']]
    
    return ds


def read_and_process_hdfs(date, 
                          base_dir,
                          target_dir,
                          dim, 
                          drop_variables=None, 
                          transform_func=None,
                          verbose=False):
    """Creates processed daily dataset by applying the transform function and concatenating
        along the specified dimension.
        
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
        
        open_kwargs = dict(decode_cf=True, decode_times=False, drop_variables=drop_variables)
        
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, **open_kwargs) as ds:
            # transform_func should do some sort of selection or aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            if ds is not None: 
                ds.load()
                return ds
            else: 
                return None
    
    date_list = date.split('-')
    day = date_list[0]
    month = date_list[1]
    year = date_list[2]
    
    paths = sorted(glob.glob(base_dir + year + '/' + year + '_' + month + '_' + day + '/' + '*.hdf'))
    if verbose:
        print(f'Processing CALIOP files for {date}')
    datasets = [process_one_path(p) for p in paths]
    datasets = [ds for ds in datasets if ds]
    
    if len(datasets):

        combined = xr.concat(datasets, dim)
        # this is to make sure time var is decoded
        combined = xr.decode_cf(combined)
        combined.to_netcdf(target_dir + year + '/' + month + '/' + "Processed_CAL_L1_Standard-V4-10_" + day + '_' +
                           month + '_' + year + ".nc")
        return combined
    
    else:
        print(f'No valid data for {date}!')