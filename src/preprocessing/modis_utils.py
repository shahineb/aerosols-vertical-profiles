import sys
import glob
import numpy as np
from pyhdf.SD import SD, SDC
import pandas as pd
from deco import concurrent, synchronized
from shapely.geometry import box, Point, Polygon

@concurrent    
def does_modis_swath_intersect(filename, target_region):
    modis_data = SD(filename, SDC.READ)
    latitude = modis_data.select('Latitude').get().astype(float)
    longitude = modis_data.select('Longitude').get().astype(float)
    
    # create point objects from latitude, longitude
    points = [Point(coord) for coord in list(zip(longitude.flatten(), latitude.flatten()))]
    
    # find points within the target region 
    valid = np.array([pt.within(target_region) for pt in points])
    if len(np.where(valid)[0]): 
        return filename 

@synchronized
def get_matching_files_modis(filenames, target_region):
    files = [None] * len(filenames)
    
    for i, filename in enumerate(filenames):
        files[i] = does_modis_swath_intersect(filename, target_region)
        
    return [x for x in files if x]