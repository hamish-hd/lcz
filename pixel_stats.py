import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import windows
from rasterstats import zonal_stats
import numpy

# Step 1: Import geotiff
geotiff_path = '/data/gedi/lcz/filled/2001_lcz.tif'
geotiff = rasterio.open(geotiff_path)

# Step 2: Read shapefile
shapefile_path = '/data/gedi/h/pcmc/OneDrive_1_19-1-2024/PCMC Ward Map.shp'
gdf = gpd.read_file(shapefile_path)

# Step 3: Convert both geotiff and shapefile to epsg 4326
gdf = gdf.to_crs(epsg=4326)
#geotiff = geotiff.to_crs(epsg=4326)

# Step 4-7: Zonal statistics for each polygon feature
for index, row in gdf.iterrows():
    # Extract geometry of the current feature
    geometry = row['geometry']
    
    # Mask the geotiff with the current polygon
    out_image, out_transform = mask(geotiff, [geometry], crop=True)

    # Get the unique pixel values and their counts (excluding -9999)
    unique_values, counts = numpy.unique(out_image[out_image != -9999], return_counts=True)

    # Create columns for each pixel value and store the count in the corresponding column
    for value, count in zip(unique_values, counts):
        if value != -9999:
            column_name = f'Pixel_{(value)}'
            gdf.at[index, column_name] = count

# Display the resulting GeoDataFrame with added columns
print(gdf)
gdf.to_csv('/data/gedi/lcz/outputs/2001_ward_lcz.csv')
#%%

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Step 1: Import geotiff
#geotiff_path = 'path/to/your/geotiff.tif'
geotiff = rasterio.open(geotiff_path)

# Step 2: Read shapefile
#shapefile_path = 'path/to/your/shapefile.shp'
gdf = gpd.read_file(shapefile_path)

# Step 3: Convert shapefile to epsg 4326
gdf = gdf.to_crs(epsg=4326)

# Display the geotiff and shapefile overlaid on it
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the geotiff
im = ax.imshow(geotiff.read(1), cmap='viridis', extent=[geotiff.bounds.left, geotiff.bounds.right, geotiff.bounds.bottom, geotiff.bounds.top], vmin=1, vmax=17)

# Plot the shapefile
gdf.plot(ax=ax, facecolor='none', edgecolor='red')

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax, label='Geotiff Pixel Values')

# Add title and labels
ax.set_title('Geotiff and Shapefile Overlay')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()
#%%
import numpy as np
geotiff_data = geotiff.read(1)
geotiff_data[np.isnan(geotiff_data)] = 0

def get_pixel_counts(geometry):
    # Mask the geotiff with the current polygon
    masked_data, _ = mask(geotiff, [geometry], crop=True, nodata=np.nan)

    # Flatten the 2D array and remove NaN values
    pixel_values = masked_data.flatten()
    pixel_values = pixel_values[~np.isnan(pixel_values)]

    # Count unique pixel values
    unique_values, counts = np.unique(pixel_values, return_counts=True)

    # Create a dictionary with unique values as keys and counts as values
    result_dict = dict(zip(unique_values, counts))
    
    return result_dict

# Apply the function to each polygon and store the results in a new column
gdf['PixelCounts'] = gdf['geometry'].apply(get_pixel_counts)

# Display the GeoDataFrame with the added column
print(gdf[['geometry', 'PixelCounts']])