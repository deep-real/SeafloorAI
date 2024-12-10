import os
import numpy as np
import rasterio
from patchify import patchify, unpatchify
from scipy import interpolate, ndimage
from scipy.ndimage import convolve

def create_patches_with_coords(src, patch_size):
    """Generate patches from the raster array and create a 2D array of geographic coordinates for each patch."""
    patches = []
    patches_coords = []

    # Iterate over the array in steps of patch_size
    for i in range(0, src.height, patch_size):
        for j in range(0, src.width, patch_size):
            if i + patch_size <= src.height and j + patch_size <= src.width: # stop condition
                # Read patch
                patch = src.read(window=rasterio.windows.Window(j, i, patch_size, patch_size))
                
                # Create 2D array for coordinates
                coords_array = np.empty((patch_size, patch_size, 2), dtype=float)
                
                # Populate the coordinates array
                for row in range(patch_size):
                    for col in range(patch_size):
                        x, y = src.xy(i + row, j + col)  # Get geographic coordinates for each pixel
                        coords_array[row, col] = [x, y]
                
                patches.append(patch)
                patches_coords.append(coords_array)
    
    return patches, patches_coords

def create_lon_lat_arrays(src):
    """Create full-resolution longitude and latitude arrays for the raster using vectorized operations."""
    # Generate grid of row and column indices
    cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
    
    # Transform pixel coordinates to geographic coordinates (longitude, latitude)
    # src.transform * (cols, rows) applies the affine transformation to the grid
    lon, lat = src.transform * (cols, rows)

    return lon, lat

def patchify_arrays(
        bs_data, bathy_data, slope_data, rugo_data, 
        sed_data, pzone_data, fault_data, fold_data, habitat_data,
        lon_array, lat_array, patch_size):
    """Patchify bs_data and coordinate arrays."""
    # mapping layers
    patches_sed_data = None
    patches_pzone_data = None
    patches_fault_data = None
    patches_fold_data = None
    patches_habitat_data = None

    # input layers
    patches_bs_data = patchify(bs_data, (patch_size, patch_size), step=patch_size//4)
    patches_bathy_data = patchify(bathy_data, (patch_size, patch_size), step=patch_size//4)
    patches_slope_data = patchify(slope_data, (patch_size, patch_size), step=patch_size//4)
    patches_rugo_data = patchify(rugo_data, (patch_size, patch_size), step=patch_size//4)
    patches_lon = patchify(lon_array, (patch_size, patch_size), step=patch_size//4)
    patches_lat = patchify(lat_array, (patch_size, patch_size), step=patch_size//4)

    if sed_data is not None:
        patches_sed_data = patchify(sed_data, (patch_size, patch_size, 16), step=patch_size//4)
        patches_sed_data = np.squeeze(patches_sed_data, axis=2)

    if pzone_data is not None:
        patches_pzone_data = patchify(pzone_data, (patch_size, patch_size, 21), step=patch_size//4)
        patches_pzone_data = np.squeeze(patches_pzone_data, axis=2)

    if fault_data is not None:
        patches_fault_data = patchify(fault_data, (patch_size, patch_size), step=patch_size//4)

    if fold_data is not None:
        patches_fold_data = patchify(fold_data, (patch_size, patch_size), step=patch_size//4)

    if habitat_data is not None:
        patches_habitat_data = patchify(habitat_data, (patch_size, patch_size, 9), step=patch_size//4)
        patches_habitat_data = np.squeeze(patches_habitat_data, axis=2)
        
    return patches_bs_data, patches_bathy_data, patches_slope_data, patches_rugo_data,\
            patches_sed_data, patches_pzone_data, patches_fault_data, patches_fold_data, patches_habitat_data, \
            patches_lon, patches_lat

def detect_and_resolve_multiple_annotations(mask):
    """
    Detects pixels with multiple annotations and resolves them by using the most frequent category among adjacent pixels for masks of shape HxWxC.

    Args:
    mask (np.array): A numpy array of shape (H, W, C) where H is the height, W is the width, and C is the number of channels/categories.

    Returns:
    np.array: A resolved mask where each pixel is assigned to a single category based on adjacency.
    """
    H, W, C = mask.shape

    # Detect multiple annotations
    annotation_sum = np.sum(mask > 0, axis=2)
    multi_annotated = annotation_sum > 1

    # If no multi-annotations, return the original mask after argmax
    if not np.any(multi_annotated):
        return np.argmax(mask, axis=2)

    # Find the most frequent category among adjacent pixels for the problematic areas
    def most_frequent_adjacent_category(data):
        # Expand mask to all categories to check adjacent values
        expanded_mask = np.zeros_like(data)
        for c in range(C):
            expanded_mask[:, :, c] = convolve(data[:, :, c] > 0, np.ones((3, 3)), mode='constant', cval=0)
        # Ignore the central pixel by subtracting its own value
        expanded_mask -= data > 0
        return np.argmax(expanded_mask, axis=2)

    # Resolve annotations based on adjacency
    resolved_categories = most_frequent_adjacent_category(mask)

    # Use original argmax where there are not multiple annotations
    argmax_categories = np.argmax(mask, axis=2)
    final_resolved_mask = np.where(multi_annotated, resolved_categories, argmax_categories)

    return final_resolved_mask


def add_background_channel(mask):
    """
    Adds an additional channel to a segmentation mask to represent background/nodata,
    where the background channel is 1 if all category channels are 0, and 0 otherwise.

    Args:
    mask (numpy.array): A numpy array of shape (H, W, C) representing the segmentation mask,
                        where C is the number of categories.

    Returns:
    numpy.array: A numpy array of shape (H, W, C+1) where the additional channel for background
                 has been added.
    """
    # Sum across the category channels
    sum_categories = np.sum(mask, axis=2)
    
    # Create the background channel where the sum is 0
    background_channel = (sum_categories == 0).astype(int)
    
    # Add the new channel as the first channel of the original mask
    new_mask = np.concatenate([background_channel[..., np.newaxis], mask], axis=2)
    
    return new_mask


def create_data(backscatter, bathy, slope, rugosity, 
                sed, pzone, fault, fold, habitat,
                lon, lat, region_name, save_dir):
    result = [] # store the data
    threshold = backscatter.shape[-2] * backscatter.shape[-1] * 0.1
    for r in range(backscatter.shape[0]):
        for c in range(backscatter.shape[1]):
            if np.sum(backscatter[r][c] == 255.0) >= threshold or \
                np.sum(bathy[r][c] == -9999) >= threshold: # invalid point --> ignore
                pass
            else:
                item = {} # create new item
                filename = f'{region_name}_{str(r).zfill(7)}_{str(c).zfill(7)}'
                item['filename'] = filename
                item['region'] = region_name

                # stack all input channels
                stacked = np.stack([backscatter[r][c], bathy[r][c], slope[r][c], rugosity[r][c],
                                    lon[r][c], lat[r][c]], axis=0)

                np.save(os.path.join(save_dir, region_name, 'input', f'{filename}.npy'), stacked)

                if sed is not None: # sediment

                    mask = add_background_channel(sed[r][c]) # H x W x C
                    mask = (mask > 0).astype(np.int8) # [0,1] masks
                    resolved_mask = detect_and_resolve_multiple_annotations(mask)

                    if np.sum(resolved_mask == 0) >= threshold:
                        item['sed'] = False
                    else:
                    # handle multiple-annotation here
                        item['sed'] = True
                        np.save(os.path.join(save_dir, region_name, 'sed', f'{filename}.npy'), resolved_mask)
                else:
                    item['sed'] = False
                
                if pzone is not None: # physio zone
                    mask = add_background_channel(pzone[r][c]) # H x W x C
                    mask = (mask > 0).astype(np.int8) # [0,1] masks
                    resolved_mask = detect_and_resolve_multiple_annotations(mask)

                    if np.sum(resolved_mask == 0) >= threshold:
                        item['pzone'] = False
                    else:
                    # handle multiple-annotation here
                        item['pzone'] = True
                        np.save(os.path.join(save_dir, region_name, 'pzone', f'{filename}.npy'), resolved_mask)
                else:
                    item['pzone'] = False

                if fault is not None: # fault
                    item['fault'] = True
                    np.save(os.path.join(save_dir, region_name, 'fault', f'{filename}.npy'), fault[r][c])
                else:
                    item['fault'] = False

                if fold is not None: # fold
                    item['fold'] = True
                    np.save(os.path.join(save_dir, region_name, 'fold', f'{filename}.npy'), fold[r][c])
                else:
                    item['fold'] = False

                if habitat is not None: # physio zone
                    mask = add_background_channel(habitat[r][c]) # H x W x C
                    mask = (mask > 0).astype(np.int8) # [0,1] masks
                    resolved_mask = detect_and_resolve_multiple_annotations(mask)

                    if np.sum(resolved_mask == 0) >= threshold:
                        item['habitat'] = False
                    else:
                    # handle multiple-annotation here
                        item['habitat'] = True
                        np.save(os.path.join(save_dir, region_name, 'habitat', f'{filename}.npy'), resolved_mask)
                else:
                    item['habitat'] = False
                
                result.append(item)
    return result

def interpolate_missing_pixels(image, nodata_value=255, method='cubic'):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    mask = (image == nodata_value)

    h, w = image.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=nodata_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

def interpolate_and_pad_nodata(data, nodata_value=255, footprint_size=3):
    """
    Interpolate 'nodata' values in a numpy array and pad any remaining nodata values using the nearest valid data,
    with a customizable footprint size for local averaging.
    
    Args:
    - data (numpy.ndarray): The input data array with nodata values.
    - nodata_value (int): The value that indicates 'nodata'.
    - footprint_size (int): Size of the square footprint for local operations.
    
    Returns:
    - numpy.ndarray: The array with nodata values interpolated and padded.
    """
    if np.isnan(nodata_value):
        mask = np.isnan(data)
    else:
        mask = data == nodata_value

    # Create a square footprint of given size
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)

    # Function to replace nodata values within the footprint
    def fill_nodata(values):
        central_value = values[len(values) // 2]

        if np.isnan(nodata_value):
            if not np.isnan(central_value):
                return central_value
            else:
                valid_values = values[~np.isnan(central_value)]
                return np.mean(valid_values) if valid_values.size > 0 else nodata_value
        else:
            if central_value != nodata_value:
                return central_value
            else:
                valid_values = values[values != nodata_value]
                return np.mean(valid_values) if valid_values.size > 0 else nodata_value

    # Interpolate using a generic filter with a custom footprint
    interpolated_data = ndimage.generic_filter(data, fill_nodata, footprint=footprint, mode='nearest')

    # Address any remaining nodata values using nearest neighbor interpolation

    if np.isnan(nodata_value):
        if np.any(np.isnan(interpolated_data)):
            indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
            interpolated_data = data[tuple(indices)]
    else:
        if np.any(interpolated_data == nodata_value):
            indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
            interpolated_data = data[tuple(indices)]

    return interpolated_data

def remove_speckle_noise(patch, filter_size=3):
    filtered_patch= ndimage.median_filter(patch, size=filter_size)
    return filtered_patch