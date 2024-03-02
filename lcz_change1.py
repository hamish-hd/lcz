import rasterio

# Open the GeoTIFFs
with rasterio.open('/data/gedi/lcz/filled/2001_lcz_r.tif') as src1, \
     rasterio.open('/data/gedi/lcz/filled/2011_lcz_r.tif') as src2, \
     rasterio.open('/data/gedi/lcz/filled/2023_lcz_r.tif') as src3:
    
    # Get the geotransform parameters for each GeoTIFF
    transform1 = src1.transform
    transform2 = src2.transform
    transform3 = src3.transform
    
    # Check if the geotransform parameters are equal for all GeoTIFFs
    if transform1 == transform2 == transform3:
        print("The pixels of all three GeoTIFFs are aligned with each other pixel by pixel.")
    else:
        print("The pixels of the GeoTIFFs are not aligned.")

#%%
import rasterio
from rasterio.enums import Resampling

# Open the GeoTIFFs
with rasterio.open('/data/gedi/lcz/filled/2001_lcz.tif') as src1, \
     rasterio.open('/data/gedi/lcz/filled/2011_lcz.tif') as src2, \
     rasterio.open('/data/gedi/lcz/filled/2023_lcz.tif') as src3:
    
    # Read the raster data
    data1 = src1.read(1)  # Assuming single-band GeoTIFFs
    data2 = src2.read(1)
    data3 = src3.read(1)
    
    # Get the affine transformation
    transform1 = src1.transform
    transform2 = src2.transform
    transform3 = src3.transform
    
    # Resample the GeoTIFFs to a common grid using the affine transformation of the first GeoTIFF
    with rasterio.open(
        '/data/gedi/lcz/filled/2001_lcz_r.tif',
        'w',
        driver='GTiff',
        height=data1.shape[0],
        width=data1.shape[1],
        count=1,
        dtype=data1.dtype,
        crs=src1.crs,
        transform=transform1,
    ) as dst1:
        dst1.write(data1, 1)
    
    with rasterio.open(
        '/data/gedi/lcz/filled/2011_lcz_r.tif',
        'w',
        driver='GTiff',
        height=data1.shape[0],  # Use the shape of the first GeoTIFF
        width=data1.shape[1],
        count=1,
        dtype=data1.dtype,
        crs=src1.crs,
        transform=transform1,  # Use the affine transformation of the first GeoTIFF
    ) as dst2:
        dst2.write(data2, 1)
    
    with rasterio.open(
        '/data/gedi/lcz/filled/2023_lcz_r.tif',
        'w',
        driver='GTiff',
        height=data1.shape[0],  # Use the shape of the first GeoTIFF
        width=data1.shape[1],
        count=1,
        dtype=data1.dtype,
        crs=src1.crs,
        transform=transform1,  # Use the affine transformation of the first GeoTIFF
    ) as dst3:
        dst3.write(data3, 1)

#%%
from rasterio.plot import show

# Open the aligned GeoTIFFs
with rasterio.open('/data/gedi/lcz/filled/2001_lcz_r.tif') as src1, \
     rasterio.open('/data/gedi/lcz/filled/2011_lcz_r.tif') as src2, \
     rasterio.open('/data/gedi/lcz/filled/2023_lcz_r.tif') as src3:
    
    # Plot each aligned GeoTIFF
    show((src1, 1), title='First Aligned GeoTIFF')
    show((src2, 1), title='Second Aligned GeoTIFF')
    show((src3, 1), title='Third Aligned GeoTIFF')
    
#%%
import rasterio
import numpy as np
import pandas as pd

# Define the paths to the input images
image_pairs = [('/data/gedi/lcz/filled/2001_lcz_r.tif', '/data/gedi/lcz/filled/2011_lcz_r.tif'), ('/data/gedi/lcz/filled/2011_lcz_r.tif', '/data/gedi/lcz/filled/2023_lcz_r.tif')]
#%%
raster_paths = ['/data/gedi/lcz/filled/2001_lcz_r.tif', '/data/gedi/lcz/filled/2011_lcz_r.tif', '/data/gedi/lcz/filled/2023_lcz_r.tif']

def get_raster_info(raster_path):
    with rasterio.open(raster_path) as src:
        # Get the data type of the pixel values
        dtype = src.dtypes[0]
        
        # Get the range of pixel values
        min_val = src.read(1).min()
        max_val = src.read(1).max()
        
        # Get any other relevant metadata
        metadata = src.meta
    
    return dtype, min_val, max_val, metadata

# Iterate through raster paths and get information
for path in raster_paths:
    dtype, min_val, max_val, metadata = get_raster_info(path)
    print(f"Raster File: {path}")
    print(f"Data Type: {dtype}")
    print(f"Range of Pixel Values: {min_val} to {max_val}")
    print("Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("\n")
#%%

land_classes = np.arange(1, 18)


change_statistics = {i: {j: 0 for j in land_classes if j != i} for i in land_classes}



def calculate_change_statistics(image1_path, image2_path):
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Read the pixel values as numpy arrays and convert to integers
        data1 = src1.read(1)
        data2 = src2.read(1)
        
        # Convert unexpected values to nan
        data1 = np.where(np.isin(data1, land_classes), data1, np.nan)
        data2 = np.where(np.isin(data2, land_classes), data2, np.nan)
        
        data1[np.isnan(data1)] = 0
        data2[np.isnan(data2)] = 0
        
        # Convert pixel values to integers
        data1 = data1.astype(int)
        data2 = data2.astype(int)
        
        # Find pixels where land type classes changed
        changed_pixels = np.where((data1 != data2) & (data1 != 0) & (data2 != 0))
        
        # Update change statistics
        for row, col in zip(*changed_pixels):
            old_class = int(data1[row, col])  # Ensure integer value
            new_class = int(data2[row, col])  # Ensure integer value
            change_statistics[old_class][new_class] += 1

# Calculate change statistics for each pair of images
calculate_change_statistics('/data/gedi/lcz/filled/2001_lcz_r.tif', '/data/gedi/lcz/filled/2011_lcz_r.tif')

# Convert the change statistics dictionary to a DataFrame
df = pd.DataFrame(change_statistics)
# Export the DataFrame to a CSV file
#%%
df.to_csv('/data/gedi/lcz/filled/land_class_change_statistics_2001_2011.csv', index_label='Old Class')

print("Change statistics exported to land_class_change_statistics_2001_2011.csv")
