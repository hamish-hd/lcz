import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask

# Define the land type classes (1 to 17)
land_classes = np.arange(1, 18)

# Placeholder class for unexpected pixel values
placeholder_class = -1

# Function to calculate change statistics for each polygon
def calculate_change_statistics(polygon, image1_path, image2_path):
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Mask raster images by polygon
        mask1, _ = mask(src1, [polygon.geometry], crop=True, nodata=np.nan)
        mask2, _ = mask(src2, [polygon.geometry], crop=True, nodata=np.nan)

        # Convert pixel values to integers
        data1 = mask1[0].astype(int)
        data2 = mask2[0].astype(int)

        # Convert unexpected values to placeholder class
        data1[~np.isin(data1, land_classes)] = placeholder_class
        data2[~np.isin(data2, land_classes)] = placeholder_class

        # Find pixels where land type classes changed and skip pixels with placeholder class
        changed_pixels = np.where((data1 != data2) & (data1 != placeholder_class) & (data2 != placeholder_class))

        # Initialize change statistics dictionary
        change_statistics = {i: {j: 0 for j in land_classes if j != i} for i in land_classes}

        # Update change statistics
        for row, col in zip(*changed_pixels):
            old_class = int(data1[row, col])
            new_class = int(data2[row, col])
            change_statistics[old_class][new_class] += 1

        return change_statistics

# Load shapefile containing polygon features
shapefile_path = 'file:///data/gedi/h/pcmc/OneDrive_1_19-1-2024/PCMC Ward Map.shp'
gdf = gpd.read_file(shapefile_path)

# Define paths to input raster files
image1_path = '/data/gedi/lcz/filled/2001_lcz_r.tif'
image2_path = '/data/gedi/lcz/filled/2011_lcz_r.tif'

# Create a directory to store individual CSV files
output_dir = '/data/gedi/lcz/filled/individual_change_statistics'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each polygon feature and calculate change statistics
for index, row in gdf.iterrows():
    # Calculate change statistics for the current polygon feature
    change_statistics = calculate_change_statistics(row, image1_path, image2_path)

    # Convert the change statistics dictionary to a DataFrame
    df = pd.DataFrame(change_statistics)

    # Sort columns in ascending order
    df = df.reindex(sorted(df.columns), axis=1)
    df.iloc[:, 1:] *= 900


    # Define the output CSV file path for the current polygon feature
    output_csv_path = os.path.join(output_dir, f'2001_2011_{row["Name"]}.csv')

    # Export the DataFrame to a CSV file
    df.to_csv(output_csv_path, index_label='Old Class')

    print(f"Change statistics for polygon {row['Name']} exported to {output_csv_path}")
