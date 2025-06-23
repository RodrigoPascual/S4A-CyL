import os
import numpy as np
import pandas as pd
import rasterio

# 1) List of years
years = [2020, 2021, 2022, 2023, 2024]

# 2) Loop through the updated rasters (labelsYYYY_unif.tif) to extract used codes
used_codes = set()
for y in years:
    new_raster = f"labels{y}_unif.tif"
    if not os.path.isfile(new_raster):
        print(f"{new_raster} not found, skipping.")
        continue
    with rasterio.open(new_raster) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            used_codes.update(np.unique(arr))

print(f"Codes used in the '_unif' rasters: {len(used_codes)} unique codes.\n")

# 3) Read the crop_id_mapping_SIGPAC_unif.csv (created in the previous step)
df_map = pd.read_csv("crop_id_mapping_SIGPAC_unif.csv", header=None, names=["label", "code"])

# 4) Keep only the rows whose codes are in used_codes
df_filtered = df_map[df_map["code"].isin(used_codes)].copy()

# Remove potential duplicates and sort by label (optional)
df_filtered.drop_duplicates(subset=["label", "code"], inplace=True)
df_filtered.sort_values(by="label", inplace=True)
df_filtered.reset_index(drop=True, inplace=True)

# 5) Reassign codes incrementally from 1 to N
df_filtered["new_code"] = range(1, len(df_filtered) + 1)

# Create a dictionary: old_code -> new_code
old_to_new = dict(zip(df_filtered["code"], df_filtered["new_code"]))

# 6) Save a final CSV with [label, new_code] columns
df_filtered[["label", "new_code"]].to_csv("crop_id_mapping.csv", index=False, header=False)
print("'crop_id_mapping.csv' has been created with the renumbered codes.\n")

# 7) Update the rasters again, going from labelsYYYY_unif.tif to labelsYYYY.tif, applying the new incremental numbering.
for y in years:
    in_raster = f"labels{y}_unif.tif"
    out_raster = f"labels{y}.tif"
    if not os.path.isfile(in_raster):
        continue

    print(f"Rewriting codes in {in_raster} => {out_raster} ...")
    with rasterio.open(in_raster) as src:
        profile = src.profile.copy()
        with rasterio.open(out_raster, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                # The lambda function maps each old code to the new one, defaulting to 0 if not found
                arr_final = np.vectorize(lambda x: old_to_new.get(x, 0))(arr)
                dst.write(arr_final, 1, window=window)
    print(f"Created => {out_raster}\n")

print("Process complete!\n"
      " - 'crop_id_mapping.csv' contains a clean mapping (1..N)\n"
      " - 'labelsYYYY.tif' are the rasters with unified and renumbered codes.")
