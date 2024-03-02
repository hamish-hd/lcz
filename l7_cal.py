import rasterio
import numpy as np

f = '/data/gedi/lcz/2001_L7'

B1 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B1.TIF'
B2 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B2.TIF'
B3 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B3.TIF'
B4 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B4.TIF'
B5 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B5.TIF'
B6 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_ST_B6.TIF'
B7 = f+'/LE07_L2SP_147047_20010324_20200917_02_T1_SR_B7.TIF'

output_path = '/data/gedi/lcz/final/2001.tif'

profile = rasterio.open(B1).profile

profile.update(count=10, dtype='float32', names=['B1 - Blue', 'B2 - Green', 'B3 - Red', 'B4 - NIR', 'B5 - SWIR 1', 'B6 - Thermal', 'B7 - SWIR 2', 'NDVI', 'NDBI','NDWI'])

def l7_cal(a):
    b = rasterio.open(a).read(1) * 0.0000275 - 0.2
    b[b == -0.2] = np.nan
    return b

b1 = l7_cal(B1)
b2 = l7_cal(B2)
b3 = l7_cal(B3)
b4 = l7_cal(B4)
b5 = l7_cal(B5)
b6 = (0.00341802*rasterio.open(B6).read(1)+149).astype('float32')
b6[b6 == -0.2] = np.nan

b7 = l7_cal(B7)

# NDVI calculation
ndvi = (b4 - b3) / (b4 + b3)
ndbi = (b5 - b4) / (b5 + b4)
ndwi = (b2 - b4) / (b2 + b4)

profile['IMAGE_STRUCTURE'] = {'INTERLEAVE': 'BAND', 'BAND_NAMES': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'NDBI', 'NDWI']}
profile['TIFFTAG_IMAGEDESCRIPTION'] = 'B1,B2,B3,B4,B5,B6,B7,NDVI,NDBI,NDWI'


# Assuming you want to write the calibrated bands and NDVI to the output GeoTIFF
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(b1, 1)
    dst.write(b2, 2)
    dst.write(b3, 3)
    dst.write(b4, 4)
    dst.write(b5, 5)
    dst.write(b6, 6)
    dst.write(b7, 7)
    dst.write(ndvi, 8)
    dst.write(ndbi,9)
    dst.write(ndwi,10)
    
