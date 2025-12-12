



import rasterio
import numpy as np

# Open the source TIFF file for reading
with rasterio.open('testslope.tif') as src:
    # Copy metadata for the new file
    meta = src.meta.copy()

    # Ensure the file has at least 4 bands
    if src.count < 4:
        raise ValueError("The TIFF file does not have a 4th band.")

    # Read all bands into a list
    bands = [src.read(i) for i in range(1, src.count + 1)]

# Replace all cell values in the 4th band with 104
bands[3][:] = 104  # band index 3 corresponds to band 4

# Define the new file name
new_filename = 'testslope_modified.tif'

# Write the updated bands to the new file
with rasterio.open(new_filename, 'w', **meta) as dst:
    for i, band_data in enumerate(bands, start=1):
        dst.write(band_data, i)

print("New file saved as", new_filename)
