import json
import numpy as np
import rasterio
from data_utils import create_data, create_lon_lat_arrays, patchify_arrays
import time

patch_size = 224
OUTPUT_DIR = "..." # directory of the dataset

# below is a sample

massachusetts = [
    {
        'name': 'region5',
        'bs_raster': './data/4326/Massachusetts/Region1/bscatter_10m_synced.tif',
        'bathy_raster': './data/4326/Massachusetts/Region1/bathy_10m_synced.tif',
        'slope_raster': './data/4326/Massachusetts/Region1/slope_10m_synced.tif',
        'rugosity_raster': './data/4326/Massachusetts/Region1/rugosity_10m_synced.tif',
        'sed_raster': './data/4326/Massachusetts/Region1/sediment_10m_synced.tif',
        'pzone_raster': './data/4326/Massachusetts/Region1/pzones_10m_synced.tif',
    },

    {   
        'name': 'region6',
        'bs_raster': './data/4326/Massachusetts/Region2/bscatter_5m_synced.tif',
        'bathy_raster': './data/4326/Massachusetts/Region2/bathy_5m_synced.tif',
        'slope_raster': './data/4326/Massachusetts/Region2/slope_5m_synced.tif',
        'rugosity_raster': './data/4326/Massachusetts/Region2/rugosity_5m_synced.tif',
        'sed_raster': './data/4326/Massachusetts/Region2/sediment_5m_synced.tif',
        'pzone_raster': './data/4326/Massachusetts/Region2/pzones_5m_synced.tif',
    },

    {
        'name': 'region7',
        'bs_raster': './data/4326/Massachusetts/Region3/bscatter_1m_synced.tif',
        'bathy_raster': './data/4326/Massachusetts/Region3/bathy_1m_synced.tif',
        'slope_raster': './data/4326/Massachusetts/Region3/slope_1m_synced.tif',
        'rugosity_raster': './data/4326/Massachusetts/Region3/rugosity_1m_synced.tif',
        'sed_raster': './data/4326/Massachusetts/Region3/sediment_1m_synced.tif',
        'pzone_raster': './data/4326/Massachusetts/Region3/pzones_1m_synced.tif',
    }
]

all_data = []
for region in massachusetts:

    print(f"----------------Processing {region['name']}----------------")

    backscatter_raster = region['bs_raster']
    bathy_raster = region['bathy_raster']
    slope_raster = region['slope_raster']
    rugosity_raster = region['rugosity_raster']

    sediment_raster = region['sed_raster']
    pzone_raster = region['pzone_raster']

    with rasterio.open(backscatter_raster, 'r') as bs,\
        rasterio.open(bathy_raster, 'r') as bathy,\
        rasterio.open(slope_raster, 'r') as slp,\
        rasterio.open(rugosity_raster, 'r') as rugo,\
        rasterio.open(sediment_raster, 'r') as sed,\
        rasterio.open(pzone_raster, 'r') as pzone:

        print(bs.height, bs.width)
        print(bathy.height, bathy.width)
        print(slp.height, slp.width)
        print(rugo.height, rugo.width)
        print(sed.height, sed.width)
        print(pzone.height, pzone.width)

        # using patchify
        print("Creating long/lat arrays...")
        start_time = time.time()
        lon_array, lat_array = create_lon_lat_arrays(bs)
        print("--- %s seconds ---" % (time.time() - start_time))

        bs_data = bs.read(1)
        bathy_data = bathy.read(1)
        slp_data = slp.read(1)
        rugo_data = rugo.read(1)

        sed_data = sed.read() # n_classes x H x W 
        sed_data = np.moveaxis(sed_data, 0, -1) # H x W x n_classes
        pzone_data = pzone.read() # n_classes x H x W
        pzone_data = np.moveaxis(pzone_data, 0, -1) # H x W x n_classes
        
        print("Patchifying...")
        start_time = time.time()

        patches_bs_data, patches_bathy_data, patches_slope_data, patches_rugo_data,\
        patches_sed_data, patches_pzone_data, _, _, _, patches_lon, patches_lat = patchify_arrays(
            bs_data, bathy_data, slp_data, rugo_data,
            sed_data, pzone_data, None, None, None,
            lon_array, lat_array, patch_size)
        
        print(patches_bs_data.shape, patches_bathy_data.shape, 
              patches_slope_data.shape, patches_rugo_data.shape,
              patches_lon.shape, patches_lat.shape,
              patches_sed_data.shape, patches_pzone_data.shape)
        print("--- %s seconds ---" % (time.time() - start_time))

        print("Creating data...")
        start_time = time.time()
        result = create_data(patches_bs_data, patches_bathy_data, 
                patches_slope_data, patches_rugo_data, 
                patches_sed_data, patches_pzone_data, 
                None, None, None,
                patches_lon, patches_lat, 
                region['name'], OUTPUT_DIR)
        print("--- %s seconds ---" % (time.time() - start_time))

        all_data += result
        
to_save = {}
to_save['anno'] = all_data
with open('sample_data.json', 'w') as outfile:
    json.dump(to_save, outfile, indent=4)

print(len(to_save['anno']))