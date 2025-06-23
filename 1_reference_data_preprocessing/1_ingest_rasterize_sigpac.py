"""
This script performs the following actions:
1) Reads one or more enclosure shapefiles (concatenating them if there are several).
2) Prints information about geometries and columns.
3) Interactively asks for the parcel ID field and the crop field.
   - If 'S4A' is entered, it uses a persistent dictionary (CSV file: parcel_id_mapping.csv)
     that maps (C_PROV_CAT, C_MUNI_CAT, C_POLIGONO, C_PARCELA, C_RECINTO) -> unique ID,
     ensuring the same ID is preserved each year.
4) If the ID field is not numeric (or does not exist), it creates an auto-incrementing "ID_PARCELA" field.
5) Reprojects the resulting shapefile to EPSG:32630.
6) Creates a "CULTIVO_CODE" field by copying the original crop field and, from it,
   assigns a persistent numeric code using a CSV (crop_id_mapping_SIGPAC.csv) so that each
   crop has a unique code.
7) Rasterizes to 'parcels.tif' (using the ID field) and 'labels.tif' (using the numeric crop code) at 10m x 10m resolution.
8) Saves/updates the CSV files for persistent IDs and crop codes.
"""

import os
import csv
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import from_origin

# --------------------------------------------------------------------
# Parcel ID persistence
# --------------------------------------------------------------------
parcel_id_map = {}  # (province, municipality, polygon, parcel, enclosure) -> ID
next_id = 1

def load_parcel_map_csv(csv_path="parcel_id_mapping.csv"):
    """Loads the parcel ID mapping from a CSV file."""
    global parcel_id_map, next_id
    if os.path.isfile(csv_path):
        print(f"✓ Reading persistent ID CSV: {csv_path}")
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                prov, muni, pol, parc, rec, assigned_id = row
                key = (int(prov), int(muni), int(pol), int(parc), int(rec))
                assigned_id = int(assigned_id)
                parcel_id_map[key] = assigned_id
                if assigned_id >= next_id:
                    next_id = assigned_id + 1
        print(f"  Loaded {len(parcel_id_map)} records. Next available ID = {next_id}.")
    else:
        print(f"✗ {csv_path} does not exist. It will be created if you use 'S4A'.")

def save_parcel_map_csv(csv_path="parcel_id_mapping.csv"):
    """Saves the parcel ID mapping to a CSV file."""
    global parcel_id_map
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for (prov, muni, pol, parc, rec), unique_id in parcel_id_map.items():
            writer.writerow([prov, muni, pol, parc, rec, unique_id])
    print(f"✓ Saved ID CSV to {csv_path}. Total {len(parcel_id_map)} rows.")

# --------------------------------------------------------------------
# Crop code persistence
# --------------------------------------------------------------------
cultivo_map = {}  # key: crop label (text), value: numeric code
next_crop_code = 1

def load_cultivo_map_csv(csv_path="crop_id_mapping_SIGPAC.csv"):
    """Loads the crop code mapping from a CSV file."""
    global cultivo_map, next_crop_code
    if os.path.isfile(csv_path):
        print(f"✓ Reading crop code CSV: {csv_path}")
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                crop_label, code = row
                code = int(code)
                cultivo_map[crop_label] = code
                if code >= next_crop_code:
                    next_crop_code = code + 1
        print(f"  Loaded {len(cultivo_map)} codes. Next available code = {next_crop_code}.")
    else:
        print(f"✗ {csv_path} does not exist. It will be created if new crops are added.")

def save_cultivo_map_csv(csv_path="crop_id_mapping_SIGPAC.csv"):
    """Saves the crop code mapping to a CSV file."""
    global cultivo_map
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for crop_label, code in cultivo_map.items():
            writer.writerow([crop_label, code])
    print(f"✓ Saved crop code CSV to {csv_path}. Total {len(cultivo_map)} rows.")

def get_or_create_s4a_id(row):
    """Gets an existing S4A ID or creates a new one for a given row."""
    global next_id, parcel_id_map
    key = (
        int(row["C_PROV_CAT"]),
        int(row["C_MUNI_CAT"]),
        int(row["C_POLIGONO"]),
        int(row["C_PARCELA"]),
        int(row["C_RECINTO"])
    )
    if key in parcel_id_map:
        return parcel_id_map[key]
    else:
        assigned_id = next_id
        parcel_id_map[key] = assigned_id
        next_id += 1
        return assigned_id

def main():
    """Main function to run the processing pipeline."""
    load_parcel_map_csv("parcel_id_mapping.csv")
    load_cultivo_map_csv("crop_id_mapping_SIGPAC.csv")

    shapefile_paths_input = input("Enter the Shapefile path (or multiple paths separated by commas): ").strip()
    shapefile_paths = [p.strip() for p in shapefile_paths_input.split(",") if p.strip()]
    if not shapefile_paths:
        print("No shapefile path specified. Exiting.")
        return

    gdfs = []
    for path in shapefile_paths:
        print("\nReading Shapefile:", path)
        gdf_temp = gpd.read_file(path)
        gdfs.append(gdf_temp)
    if len(gdfs) > 1:
        crs_base = gdfs[0].crs
        for i, gf in enumerate(gdfs[1:], start=2):
            if gf.crs != crs_base:
                print(f"WARNING: Shapefile {i} has a different CRS. It will be reprojected to match the first one...")
                gdfs[i-1] = gf.to_crs(crs_base)
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        print(f"\nConcatenated {len(gdfs)} shapefiles. Total rows: {len(gdf)}")
    else:
        gdf = gdfs[0]
        print(f"Total rows: {len(gdf)}")

    output_shapefile = "recintos_merged_32630.shp"
    epsg_target = 32630  # Target EPSG
    resolution = 10.0    # Pixel size in meters

    # 1) Basic geometry information
    print("\nEmpty geometries (None):", gdf.geometry.isna().sum())
    print("Empty geometries (is_empty):", gdf.geometry.is_empty.sum())
    print("Invalid geometries (is_valid=False):", (~gdf.geometry.is_valid).sum())
    print("--------------------------------------------------------------------------------")

    # 2) Correct/filter null or invalid geometries
    before_len = len(gdf)
    gdf = gdf[~gdf.geometry.isna()]
    print(f"Removed {before_len - len(gdf)} rows with geometry=None.")
    before_len = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty]
    print(f"Removed {before_len - len(gdf)} rows with empty geometry.")
    inv_before = (~gdf.geometry.is_valid).sum()
    if inv_before > 0:
        gdf["geometry"] = gdf.geometry.buffer(0)
    inv_after = (~gdf.geometry.is_valid).sum()
    print(f"Invalid geometries before: {inv_before}, after applying buffer(0): {inv_after}")
    print("--------------------------------------------------------------------------------")

    # 3) Display available columns
    print("Available columns in the GeoDataFrame:")
    for col in gdf.columns:
        print("   -", col)
    print("--------------------------------------------------------------------------------")

    # 4) Interactively ask for the parcel ID field and the crop field
    print("NOTE: Type 'S4A' to use persistent IDs based on (C_PROV_CAT, C_MUNI_CAT, C_POLIGONO, C_PARCELA, C_RECINTO).")
    id_field_input = input("Parcel ID field [e.g., COD_RECINTO]: ").strip()
    cultivo_field_input = input("Crop field [e.g., D_PRODUCTO]: ").strip()
    print("--------------------------------------------------------------------------------")
    if id_field_input.lower() == "geometry":
        print("ERROR: 'geometry' is not a valid ID field. Creating auto-incrementing 'ID_PARCELA'.")
        id_field_input = ""

    # 5) Assign IDs as indicated
    needed_fields = ["C_PROV_CAT", "C_MUNI_CAT", "C_POLIGONO", "C_PARCELA", "C_RECINTO"]
    if id_field_input.upper() == "S4A":
        missing = [f for f in needed_fields if f not in gdf.columns]
        if missing:
            print(f"ERROR: Missing fields {missing} to use 'S4A'.")
            print("Creating auto-incrementing 'ID_PARCELA'...")
            gdf["ID_PARCELA"] = range(1, len(gdf) + 1)
            id_field_input = "ID_PARCELA"
        else:
            print("Assigning persistent IDs with S4A...")
            gdf["ID_S4A"] = gdf.apply(get_or_create_s4a_id, axis=1)
            id_field_input = "ID_S4A"
            print("'ID_S4A' field created successfully.")
    else:
        if not id_field_input or (id_field_input not in gdf.columns):
            print("ID field does not exist. Creating auto-incrementing 'ID_PARCELA'...")
            gdf["ID_PARCELA"] = range(1, len(gdf) + 1)
            id_field_input = "ID_PARCELA"
        else:
            if not np.issubdtype(gdf[id_field_input].dtype, np.number):
                print(f"The field '{id_field_input}' is not numeric. Creating auto-incrementing 'ID_PARCELA'...")
                gdf["ID_PARCELA"] = range(1, len(gdf) + 1)
                id_field_input = "ID_PARCELA"
            else:
                print(f"Using '{id_field_input}' as parcel ID.")

    duplicates = gdf[id_field_input].duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: There are {duplicates} duplicate IDs in the '{id_field_input}' column.")
    else:
        print(f"OK: No duplicates found in '{id_field_input}'.")
    print("--------------------------------------------------------------------------------")

    # 6) Verify that the crop field exists
    if cultivo_field_input not in gdf.columns:
        print(f"ERROR: The crop field '{cultivo_field_input}' does not exist. Aborting.")
        return
    else:
        print(f"Using '{cultivo_field_input}' as the crop field.")
    print("--------------------------------------------------------------------------------")

    # 7) Reproject the GeoDataFrame to EPSG:32630
    old_crs = gdf.crs
    print(f"Reprojecting from {old_crs} to EPSG:{epsg_target}...")
    gdf = gdf.to_crs(epsg=epsg_target)
    new_crs = gdf.crs
    gdf.to_file(output_shapefile, driver="ESRI Shapefile")
    print(f"Saved reprojected shapefile => {output_shapefile}")

    # 8) Create the CULTIVO_CODE field by copying the original crop field
    code_field = "CULTIVO_CODE"
    print(f"Copying the '{cultivo_field_input}' field to '{code_field}'.")
    gdf[code_field] = gdf[cultivo_field_input]
    gdf.to_file(output_shapefile, driver="ESRI Shapefile") # Overwrite to save the new column
    print(f"Updated shapefile with '{code_field}'.")
    print("--------------------------------------------------------------------------------")

    # 9) Assign persistent codes to each crop using cultivo_map
    # Get the list of unique crops from the GeoDataFrame
    unique_crops = gdf[code_field].unique()
    global next_crop_code
    for crop in unique_crops:
        if crop not in cultivo_map:
            cultivo_map[crop] = next_crop_code
            next_crop_code += 1
    gdf["CULTIVO_CODE_NUM"] = gdf[code_field].map(cultivo_map)
    print("Mapping from CULTIVO_CODE_NUM to original labels:")
    for label, code in cultivo_map.items():
        print(f"  {code}: {label}")
    print("--------------------------------------------------------------------------------")

    # 10) Rasterize parcels.tif using the ID field
    out_parcels = "parcels.tif"
    print(f"Generating '{out_parcels}' with the ID '{id_field_input}'...")
    minx, miny, maxx, maxy = gdf.total_bounds
    resolution = float(resolution)
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    shapes_parcels = (
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf[id_field_input])
    )
    parcels_arr = features.rasterize(
        shapes=shapes_parcels,
        out_shape=(height, width),
        fill=0,
        transform=transform,
        dtype="uint32"
    )
    parcels_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint32",
        "crs": gdf.crs.to_wkt(),
        "transform": transform,
        "compress": "deflate"
    }
    with rasterio.open(out_parcels, "w", **parcels_profile) as dst:
        dst.write(parcels_arr, 1)
    print(f"Saved => {out_parcels}")

    # 11) Rasterize labels.tif using the CULTIVO_CODE_NUM field
    out_labels = "labels.tif"
    print(f"Generating '{out_labels}' with 'CULTIVO_CODE_NUM'...")
    shapes_labels = (
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf["CULTIVO_CODE_NUM"])
    )
    labels_arr = features.rasterize(
        shapes=shapes_labels,
        out_shape=(height, width),
        fill=0,
        transform=transform,
        dtype="uint32"
    )
    labels_profile = parcels_profile.copy()
    with rasterio.open(out_labels, "w", **labels_profile) as dst:
        dst.write(labels_arr, 1)
    print(f"Saved => {out_labels}")
    print("--------------------------------------------------------------------------------")

    # 12) Save the persistence CSVs
    if id_field_input == "ID_S4A":
        save_parcel_map_csv("parcel_id_mapping.csv")
    save_cultivo_map_csv("crop_id_mapping_SIGPAC.csv")

    # 13) Final messages and summary
    print("Process completed successfully.\nSummary:")
    if len(shapefile_paths) == 1:
        print(f"  • Reprojected {shapefile_paths[0]} ({old_crs}) to {output_shapefile} ({new_crs}).")
    else:
        print(f"  • Concatenated {len(shapefile_paths)} shapefiles and reprojected to {output_shapefile} ({new_crs}).")
    print(f"  • Final ID field: '{id_field_input}'. Crop: '{cultivo_field_input}'.")
    print(f"  • Raster resolution: {resolution} m/pixel.")

    def human_size(path):
        """Returns the file size in a human-readable format."""
        b = os.path.getsize(path)
        kb = b / 1024
        mb = kb / 1024
        if mb >= 1:
            return f"{mb:.2f} MB"
        elif kb >= 1:
            return f"{kb:.2f} KB"
        else:
            return f"{b} bytes"

    print("\nGenerated files:")
    print(f"  - {output_shapefile}")
    print(f"  - {out_parcels} => {human_size(out_parcels)}")
    print(f"  - {out_labels} => {human_size(out_labels)}")
    if id_field_input == "ID_S4A":
        print("  - parcel_id_mapping.csv => Persistent ID dictionary")
    print("  - crop_id_mapping_SIGPAC.csv => Crop code dictionary")
    print("\nEnd of process!")

if __name__ == "__main__":
    main()
