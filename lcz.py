import geopandas as gpd, rasterio, pandas as pd
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from shapely.geometry import mapping
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
import itertools
#%%
src = rasterio.open('/data/gedi/lcz/filled/2021_p.tif')

gdf = gpd.read_file('file:///data/gedi/lcz/samples_20240228/samples_20240228.shp')

print(gdf.head())
#%%
gdf = gdf.to_crs(src.crs)
pixel_values_by_band = [[] for _ in range(src.count)]  # One list for each band
class_labels = []

for index, row in gdf.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [mapping(geom)], crop=True)

    for band_num in range(src.count):
        band_values = out_image[band_num, :, :].flatten()
        pixel_values_by_band[band_num].extend(band_values)

    class_labels.extend([row['layer']] * len(band_values))

columns = [f"Band_{i+1}" for i in range(src.count)]
pixel_df = pd.DataFrame(np.array(pixel_values_by_band).T, columns=columns)
pixel_df['ClassLabel'] = class_labels

print(pixel_df.head())
#%%
a = pixel_df.dropna(subset='Band_1')
a.fillna(0, inplace=True)
#%%
X = a.drop('ClassLabel', axis=1)
y = a['ClassLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#%%
# Feature Selection
feature_importance = clf.feature_importances_
important_features = X.columns[feature_importance > 0.01]  # Adjust the threshold as needed
#%%
# Select only important features
X_selected = X[important_features]

# Print selected features
print("Selected Features:", list(X_selected.columns))
#%%
# Retrain the model with selected features
X_train_selected, X_test_selected = train_test_split(X_selected, test_size=0.2, random_state=42)

clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
clf_selected.fit(X_train_selected, y_train)

# Evaluate the model with selected features on the test set
y_pred_selected = clf_selected.predict(X_test_selected)
print("Accuracy with Selected Features:", accuracy_score(y_test, y_pred_selected))
print("\nClassification Report with Selected Features:\n", classification_report(y_test, y_pred_selected))
#%%

#%%
import os
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Get the number of available cores using os.cpu_count()
num_cores = os.cpu_count()

# Use half of the available cores for parallel processing
num_cores_to_use = max(1, num_cores // 2)  # Ensure at least 1 core is used

# Hyperparameter Tuning with Half of Available Cores
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set n_jobs to use half of the available cores
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=num_cores_to_use)
grid_search.fit(X_train_selected, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Retrain the model with the best parameters
clf_tuned = RandomForestClassifier(random_state=42, **best_params)
clf_tuned.fit(X_train_selected, y_train)

# Evaluate the model with tuned hyperparameters on the test set
y_pred_tuned = clf_tuned.predict(X_test_selected)
print("Accuracy with Tuned Hyperparameters:", accuracy_score(y_test, y_pred_tuned))
print("\nClassification Report with Tuned Hyperparameters:\n", classification_report(y_test, y_pred_tuned))
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(a.drop('ClassLabel', axis=1))
pixel_df_cleaned_scaled = pd.DataFrame(X_scaled, columns=columns)
pixel_df_cleaned_scaled['ClassLabel'] = a['ClassLabel']