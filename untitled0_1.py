import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from shapely.geometry import mapping
import joblib


# Load raster data
src = rasterio.open('/data/gedi/lcz/filled/2021_p.tif')

# Load shapefile with training samples
gdf = gpd.read_file('file:///data/gedi/lcz/samples_20240228/samples_20240228.shp')
gdf = gdf.to_crs(src.crs)
class_label_mapping = {'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17}

# Replace class labels in the 'class' column
gdf['class'] = gdf['class'].replace(class_label_mapping)

# Convert the 'class' column to integers
gdf['class'] = pd.to_numeric(gdf['class'], errors='coerce', downcast='integer')

# Extract pixel values for each sample
pixel_values_by_band = [[] for _ in range(src.count)]
class_labels = []

for index, row in gdf.iterrows():
    geom = row['geometry']
    out_image, _ = mask(src, [mapping(geom)], crop=True)

    for band_num in range(src.count):
        band_values = out_image[band_num, :, :].flatten()
        pixel_values_by_band[band_num].extend(band_values)

    class_labels.extend([row['class']] * len(band_values))

# Create DataFrame with pixel values and class labels
columns = [f"Band_{i+1}" for i in range(src.count)]
pixel_df = pd.DataFrame(np.array(pixel_values_by_band).T, columns=columns)
pixel_df['ClassLabel'] = class_labels

# Drop rows with missing values and fill NaNs with 0
pixel_df.dropna(subset=['Band_1'], inplace=True)
pixel_df.fillna(0, inplace=True)

# Split data into training and testing sets
X = pixel_df.drop('ClassLabel', axis=1)
y = pixel_df['ClassLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Kappa Coefficient:", kappa)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict class labels for the entire raster
all_pixel_values_by_band = [src.read(band_num + 1) for band_num in range(src.count)]
all_pixel_values = np.stack(all_pixel_values_by_band, axis=-1)
reshaped_pixel_values = all_pixel_values.reshape(-1, src.count)

remaining_pixels_df = pd.DataFrame(reshaped_pixel_values, columns=columns)
remaining_pixels_df.fillna(0, inplace=True)

predicted_labels = clf.predict(remaining_pixels_df)
predicted_labels_reshaped = predicted_labels.reshape(all_pixel_values.shape[0], all_pixel_values.shape[1])

# Save predicted labels to a new raster file
profile = src.profile
profile.update(count=1, dtype=rasterio.int32)
profile.update(nodata=-9999, count=1, dtype=rasterio.int32)

src_nan_mask = np.isnan(all_pixel_values[:,:,0])
predicted_labels_reshaped[src_nan_mask] = -9999

with rasterio.open('/data/gedi/lcz/filled/predicted_classes_2802_n.tif', 'w', **profile) as dst:
    dst.write(predicted_labels_reshaped.astype(rasterio.int32), 1)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

model_export_path = '/data/gedi/lcz/filled/lcz_model.pkl'  # Replace with the desired path
joblib.dump(clf, model_export_path)

#%% Predict new
'''
import joblib
import rasterio
import numpy as np

# Load the exported model
model_export_path = '/data/gedi/lcz/filled/lcz_model.pkl'  # Replace with the actual path
clf = joblib.load(model_export_path)

# Load the new raster data
new_src = rasterio.open('/data/gedi/lcz/filled/2011_p.tif')  # Replace with the actual path

# Predict class labels for the entire new raster
new_pixel_values_by_band = [new_src.read(band_num + 1) for band_num in range(new_src.count)]
new_all_pixel_values = np.stack(new_pixel_values_by_band, axis=-1)
new_reshaped_pixel_values = new_all_pixel_values.reshape(-1, new_src.count)

new_predicted_labels = clf.predict(new_reshaped_pixel_values)
new_predicted_labels_reshaped = new_predicted_labels.reshape(new_all_pixel_values.shape[0], new_all_pixel_values.shape[1])

# Save predicted labels to a new raster file
new_profile = new_src.profile
new_profile.update(count=1, dtype=rasterio.int32)
new_profile.update(nodata=-9999, count=1, dtype=rasterio.int32)

new_nan_mask = np.isnan(new_all_pixel_values[:,:,0])
new_predicted_labels_reshaped[new_nan_mask] = -9999

with rasterio.open('/data/gedi/lcz/filled/predicted_classes_2011.tif', 'w', **new_profile) as new_dst:
    new_dst.write(new_predicted_labels_reshaped.astype(rasterio.int32), 1)
'''
#%%
# Load the new raster data for 2001
new_src_2001 = rasterio.open('/data/gedi/lcz/filled/2001.tif')  # Replace with the actual path

# Predict class labels for the entire new raster (2001)
new_pixel_values_by_band_2001 = [new_src_2001.read(band_num + 1) for band_num in range(new_src_2001.count)]
new_all_pixel_values_2001 = np.stack(new_pixel_values_by_band_2001, axis=-1)
new_reshaped_pixel_values_2001 = new_all_pixel_values_2001.reshape(-1, new_src_2001.count)

# Fill NaN values with 0
new_reshaped_pixel_values_2001 = np.nan_to_num(new_reshaped_pixel_values_2001, nan=0)

new_predicted_labels_2001 = clf.predict(new_reshaped_pixel_values_2001)
new_predicted_labels_reshaped_2001 = new_predicted_labels_2001.reshape(new_all_pixel_values_2001.shape[0], new_all_pixel_values_2001.shape[1])

# Save predicted labels to a new raster file for 2001
new_profile_2001 = new_src_2001.profile
new_profile_2001.update(count=1, dtype=rasterio.int32)
new_profile_2001.update(nodata=-9999, count=1, dtype=rasterio.int32)

new_nan_mask_2001 = np.isnan(new_all_pixel_values_2001[:,:,0])
new_predicted_labels_reshaped_2001[new_nan_mask_2001] = -9999

with rasterio.open('/data/gedi/lcz/filled/predicted_classes_2001.tif', 'w', **new_profile_2001) as new_dst_2001:
    new_dst_2001.write(new_predicted_labels_reshaped_2001.astype(rasterio.int32), 1)

# Load the new raster data for 2011
new_src_2011 = rasterio.open('/data/gedi/lcz/filled/2011.tif')  # Replace with the actual path

# Predict class labels for the entire new raster (2011)
new_pixel_values_by_band_2011 = [new_src_2011.read(band_num + 1) for band_num in range(new_src_2011.count)]
new_all_pixel_values_2011 = np.stack(new_pixel_values_by_band_2011, axis=-1)
new_reshaped_pixel_values_2011 = new_all_pixel_values_2011.reshape(-1, new_src_2011.count)

new_predicted_labels_2011 = clf.predict(new_reshaped_pixel_values_2011)
new_predicted_labels_reshaped_2011 = new_predicted_labels_2011.reshape(new_all_pixel_values_2011.shape[0], new_all_pixel_values_2011.shape[1])

# Save predicted labels to a new raster file for 2011
new_profile_2011 = new_src_2011.profile
new_profile_2011.update(count=1, dtype=rasterio.int32)
new_profile_2011.update(nodata=-9999, count=1, dtype=rasterio.int32)

new_nan_mask_2011 = np.isnan(new_all_pixel_values_2011[:,:,0])
new_predicted_labels_reshaped_2011[new_nan_mask_2011] = -9999

with rasterio.open('/data/gedi/lcz/filled/predicted_classes_2011.tif', 'w', **new_profile_2011) as new_dst_2011:
    new_dst_2011.write(new_predicted_labels_reshaped_2011.astype(rasterio.int32), 1)
