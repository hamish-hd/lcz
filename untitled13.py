import rasterio
from rasterio.transform import from_origin
import numpy as np

land_use1 = rasterio.open('/data/gedi/lcz/filled/2001_lcz_r.tif').read(1)
land_use2 = rasterio.open('/data/gedi/lcz/filled/2011_lcz_r.tif').read(1)

nan_mask = np.isnan(land_use1)



# Create a mapping dictionary for pixel values in raster 1 and raster 2 to output values
mapping = {
    (1,1):0,
(2,1):2,
(3,1):4,
(4,1):4,
(5,1):4,
(6,1):4,
(7,1):6,
(8,1):6,
(9,1):6,
(10,1):6,
(11,1):6,
(12,1):6,
(13,1):6,
(14,1):6,
(15,1):6,
(16,1):6,
(17,1):5,

(1,2):2,
(2,2):0,
(3,2):2,
(4,2):4,
(5,2):4,
(6,2):4,
(7,2):4,
(8,2):6,
(9,2):6,
(10,2):6,
(11,2):6,
(12,2):6,
(13,2):6,
(14,2):6,
(15,2):6,
(16,2):6,
(17,2):5,

(1,3):3,
(2,3):2,
(3,3):0,
(4,3):2,
(5,3):4,
(6,3):4,
(7,3):4,
(8,3):4,
(9,3):6,
(10,3):6,
(11,3):6,
(12,3):6,
(13,3):6,
(14,3):6,
(15,3):6,
(16,3):6,
(17,3):5,

(1,4):3,
(2,4):3,
(3,4):2,
(4,4):0,
(5,4):2,
(6,4):4,
(7,4):4,
(8,4):4,
(9,4):4,
(10,4):6,
(11,4):6,
(12,4):6,
(13,4):6,
(14,4):6,
(15,4):6,
(16,4):6,
(17,4):5,

(1,5):3,
(2,5):3,
(3,5):3,
(4,5):2,
(5,5):0,
(6,5):2,
(7,5):4,
(8,5):4,
(9,5):4,
(10,5):4,
(11,5):6,
(12,5):6,
(13,5):6,
(14,5):6,
(15,5):6,
(16,5):6,
(17,5):5,

(1,6):3,
(2,6):3,
(3,6):3,
(4,6):3,
(5,6):2,
(6,6):0,
(7,6):2,
(8,6):4,
(9,6):4,
(10,6):4,
(11,6):4,
(12,6):6,
(13,6):6,
(14,6):6,
(15,6):6,
(16,6):6,
(17,6):5,

(1,7):3,
(2,7):3,
(3,7):3,
(4,7):3,
(5,7):3,
(6,7):2,
(7,7):0,
(8,7):2,
(9,7):4,
(10,7):4,
(11,7):4,
(12,7):4,
(13,7):6,
(14,7):6,
(15,7):6,
(16,7):6,
(17,7):5,

(1,8):3,
(2,8):3,
(3,8):3,
(4,8):3,
(5,8):3,
(6,8):3,
(7,8):2,
(8,8):0,
(9,8):2,
(10,8):4,
(11,8):4,
(12,8):4,
(13,8):4,
(14,8):6,
(15,8):6,
(16,8):6,
(17,8):5,

(1,9):5,
(2,9):5,
(3,9):5,
(4,9):5,
(5,9):3,
(6,9):3,
(7,9):3,
(8,9):3,
(9,9):0,
(10,9):2,
(11,9):2,
(12,9):2,
(13,9):2,
(14,9):2,
(15,9):2,
(16,9):2,
(17,9):5,

(1,10):5,
(2,10):5,
(3,10):5,
(4,10):5,
(5,10):5,
(6,10):5,
(7,10):5,
(8,10):5,
(9,10):5,
(10,10):0,
(11,10):6,
(12,10):6,
(13,10):6,
(14,10):6,
(15,10):6,
(16,10):6,
(17,10):5,

(1,11):5,
(2,11):5,
(3,11):5,
(4,11):5,
(5,11):5,
(6,11):5,
(7,11):5,
(8,11):5,
(9,11):5,
(10,11):5,
(11,11):0,
(12,11):6,
(13,11):6,
(14,11):6,
(15,11):6,
(16,11):6,
(17,11):5,

(1,12):5,
(2,12):5,
(3,12):5,
(4,12):5,
(5,12):5,
(6,12):5,
(7,12):5,
(8,12):5,
(9,12):5,
(10,12):5,
(11,12):5,
(12,12):0,
(13,12):1,
(14,12):1,
(15,12):1,
(16,12):1,
(17,12):5,

(1,13):5,
(2,13):5,
(3,13):5,
(4,13):5,
(5,13):5,
(6,13):5,
(7,13):5,
(8,13):5,
(9,13):5,
(10,13):5,
(11,13):5,
(12,13):1,
(13,13):0,
(14,13):1,
(15,13):1,
(16,13):1,
(17,13):5,

(1,14):5,
(2,14):5,
(3,14):5,
(4,14):5,
(5,14):5,
(6,14):5,
(7,14):5,
(8,14):5,
(9,14):5,
(10,14):5,
(11,14):5,
(12,14):1,
(13,14):1,
(14,14):0,
(15,14):1,
(16,14):1,
(17,14):5,

(1,15):5,
(2,15):5,
(3,15):5,
(4,15):5,
(5,15):5,
(6,15):5,
(7,15):5,
(8,15):5,
(9,15):5,
(10,15):5,
(11,15):5,
(12,15):1,
(13,15):1,
(14,15):1,
(15,15):0,
(16,15):1,
(17,15):5,

(1,16):5,
(2,16):5,
(3,16):5,
(4,16):5,
(5,16):5,
(6,16):5,
(7,16):5,
(8,16):5,
(9,16):5,
(10,16):5,
(11,16):5,
(12,16):1,
(13,16):1,
(14,16):1,
(15,16):1,
(16,16):0,
(17,16):5,

(1,17):5,
(2,17):5,
(3,17):5,
(4,17):5,
(5,17):5,
(6,17):5,
(7,17):5,
(8,17):5,
(9,17):5,
(10,17):5,
(11,17):5,
(12,17):5,
(13,17):5,
(14,17):5,
(15,17):5,
(16,17):5,
(17,17):0,
    # Add more mappings as needed
}

original_data = rasterio.open('/data/gedi/lcz/filled/2001_lcz_r.tif').read(1)

output_raster = np.full_like(original_data, fill_value=np.nan, dtype=np.float32)

    # Apply the mapping to create the output raster
for keys, value in mapping.items():
    mask = (original_data == keys[0]) & (land_use2 == keys[1])
    output_raster[mask] = value

output_raster[nan_mask] = np.nan
# Write the output raster to a new file
output_profile = rasterio.open('/data/gedi/lcz/filled/2011_lcz_r.tif').profile
output_profile.update(count=1, dtype=np.float32)  # Update the profile for a single-band output raster


# Replace 'path_to_output_raster' with the desired path for the output raster file.
with rasterio.open('/data/gedi/lcz/filled/2001_2011_ps.tif', 'w', **output_profile) as dst:
    dst.write(output_raster.astype(np.float32), 1)

#%%

import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import pandas as pd

# Replace 'path_to_geotiff' and 'path_to_shapefile' with the actual paths to your files.
geotiff_path = '/data/gedi/lcz/filled/2001_2011_ps.tif'
shapefile_path = 'file:///data/gedi/h/pcmc/OneDrive_1_19-1-2024/PCMC Ward Map.shp'

# Read the shapefile
shapefile_data = gpd.read_file(shapefile_path)

# Ensure both have the same projection
geotiff_crs = shapefile_data.crs

# Pixel values to count
pixel_values = [0, 1, 2, 3, 4, 5, 6]

# Initialize an empty DataFrame
results_df = pd.DataFrame(columns=["Polygon_ID", "Name"] + [f"Count_{value}" for value in pixel_values])

# Add Polygon_ID and Name to the DataFrame
results_df["Polygon_ID"] = range(1, len(shapefile_data) + 1)
results_df["Name"] = shapefile_data["Name"]

# Perform zonal statistics for each pixel value
for value in pixel_values:
    stats = zonal_stats(
        shapefile_path,
        geotiff_path,
        stats="count",
        categorical=True,
        category_map={value: value}
    )

    # Extract count for the current pixel value
    count_values = [feature.get(value, 0) for feature in stats]

    # Add the count to the results DataFrame
    results_df[f"Count_{value}"] = count_values



results_df.to_csv('/data/gedi/lcz/filled/output_stats_2001-2011.csv', index=False)
