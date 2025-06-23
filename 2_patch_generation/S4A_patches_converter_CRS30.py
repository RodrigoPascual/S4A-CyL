import os
import re
import argparse
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.windows import Window
from affine import Affine


#############################
# PARAMETERS AND CONFIGURATION
#############################

# Dimensions of each band (y, x) in Sen4AgriNet
BAND_DIMENSIONS = {
    "B01": (61, 61),
    "B09": (61, 61),
    "B10": (61, 61),

    "B02": (366, 366),
    "B03": (366, 366),
    "B04": (366, 366),
    "B08": (366, 366),

    "B05": (183, 183),
    "B06": (183, 183),
    "B07": (183, 183),
    "B8A": (183, 183),
    "B11": (183, 183),
    "B12": (183, 183),
}

# Bands and resolution in meters
BANDS_INFO = {
    "B01": ("R60m", 60),
    "B09": ("R60m", 60),
    "B10": ("R60m", 60),

    "B02": ("R10m", 10),
    "B03": ("R10m", 10),
    "B04": ("R10m", 10),
    "B08": ("R10m", 10),

    "B05": ("R20m", 20),
    "B06": ("R20m", 20),
    "B07": ("R20m", 20),
    "B8A": ("R20m", 20),
    "B11": ("R20m", 20),
    "B12": ("R20m", 20),
}

# Number of patches in X and Y (30x30 = 900 patches)
NX_PATCHES = 30
NY_PATCHES = 30

COUNTRY_CODE = "ES"  # Spain

MAX_ZERO_PERCENT_B02 = 0.02  # 2%


#############################
# HELPER FUNCTIONS
#############################

def parse_args():
    """
    Reads the lists of years and tiles from the command line.
    Example:
        python create_patches.py --years 2022 2023 --tiles 30TUL 30TVL
    """
    parser = argparse.ArgumentParser(description="Generate NetCDF patches from Sentinel, parcels, and labels.")
    parser.add_argument("--years", nargs="+", required=True,
                        help="List of years to process (e.g., 2022 2023).")
    parser.add_argument("--tiles", nargs="+", required=True,
                        help="List of tiles to process (e.g., 30TUL 30TVL).")
    parser.add_argument("--base_dir", default="Datos brutos 2023",
                        help="Base path where the Sentinel, Parcels, and Labels folders are located.")
    args = parser.parse_args()
    return args


def find_safe_dirs(sentinel_folder):
    """
    Searches for .SAFE subdirectories in the 'sentinel_folder' directory.
    Returns a list of tuples (datetime, path_SAFE),
    sorted by date.
    """
    safe_dirs = []
    if not sentinel_folder.exists():
        return []

    for item in sentinel_folder.iterdir():
        if item.is_dir() and item.suffix == ".SAFE":
            match = re.search(r"(\d{8}T\d{6})", item.name)
            if match:
                dt_str = match.group(1)
                dt = datetime.strptime(dt_str[:8], "%Y%m%d")
                safe_dirs.append((dt, item))
    safe_dirs.sort(key=lambda x: x[0])
    return safe_dirs


def load_band_path(safe_path, band_name):
    """
    Given the .SAFE folder and the band name (e.g., B02),
    returns the path to the corresponding JP2 file (or None if it does not exist).
    """
    subfolder, _ = BANDS_INFO[band_name]
    granule_dirs = list((safe_path / "GRANULE").glob("*"))
    if not granule_dirs:
        return None
    img_data_dir = granule_dirs[0] / "IMG_DATA" / subfolder
    pattern = f"*_{band_name}_*.jp2"
    matches = list(img_data_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def time_to_datetime64(dt):
    return np.datetime64(dt.isoformat())


def dt64_to_py_datetime(dt64val):
    import pandas as pd
    return pd.Timestamp(dt64val).to_pydatetime()


def read_patch_bands(raster_path, x_i, y_i, band_name):
    """
    Reads a patch (x_i, y_i) for the band 'band_name',
    using (patch_height, patch_width) from BAND_DIMENSIONS.
    Returns a numpy array with the data portion,
    or None if there is a problem with the window.
    """
    patch_height, patch_width = BAND_DIMENSIONS[band_name]
    window = Window(x_i * patch_width, y_i * patch_height, patch_width, patch_height)
    with rasterio.open(raster_path) as src:
        data = src.read(1, window=window)
        if data.shape != (patch_height, patch_width):
            return None
    return data


def compute_coords_bands(raster_path, x_i, y_i, band_name):
    """
    Calculates 1D arrays (y_vals, x_vals) for the band 'band_name'
    (of size patch_height, patch_width) in the window (x_i, y_i),
    based on the transform of the 'raster_path' raster.
    """
    patch_height, patch_width = BAND_DIMENSIONS[band_name]
    with rasterio.open(raster_path) as src:
        t = src.transform
        start_px_x = x_i * patch_width
        start_px_y = y_i * patch_height

        x_vals = []
        y_vals = []
        for col in range(patch_width):
            x_val = t.c + t.a*(start_px_x + col)
            x_vals.append(x_val)
        for row in range(patch_height):
            y_val = t.f + t.e*(start_px_y + row)
            y_vals.append(y_val)

    return np.array(y_vals, dtype=np.float64), np.array(x_vals, dtype=np.float64)


def read_patch_same_area(ref_raster_path, x_i, y_i, ref_band, target_raster_path):
    """
    Reads from 'target_raster_path' the SAME geographical area
    that corresponds to the patch (x_i,y_i) in the 'ref_band' band of 'ref_raster_path'.

    1) geospatial bounding box in ref_raster
    2) inverse transform in target_raster
    3) read with Window(...) => clips/overwrites to (366x366) if the reference patch is B02.
    """
    patch_height, patch_width = BAND_DIMENSIONS[ref_band]

    with rasterio.open(ref_raster_path) as ref_src:
        ref_transform = ref_src.transform

        start_x = x_i * patch_width
        start_y = y_i * patch_height

        x0 = ref_transform.c + ref_transform.a*start_x
        y0 = ref_transform.f + ref_transform.e*start_y
        x1 = ref_transform.c + ref_transform.a*(start_x + patch_width - 1)
        y1 = ref_transform.f + ref_transform.e*(start_y + patch_height - 1)

        minx, maxx = sorted([x0, x1])
        miny, maxy = sorted([y0, y1])

    with rasterio.open(target_raster_path) as tgt:
        inv_t = ~tgt.transform

        px0, py0 = inv_t*(minx, maxy)
        px1, py1 = inv_t*(maxx, miny)

        px0i, py0i = int(np.floor(px0)), int(np.floor(py0))
        px1i, py1i = int(np.ceil(px1)), int(np.ceil(py1))

        w = px1i - px0i + 1
        h = py1i - py0i + 1

        window = Window(px0i, py0i, w, h)
        data = tgt.read(1, window=window)

        out = np.zeros((patch_height, patch_width), data.dtype)

        min_h = min(h, patch_height)
        min_w = min(w, patch_width)

        out[0:min_h, 0:min_w] = data[0:min_h, 0:min_w]

    return out


def read_time_series_of_patch_B02(safe_list, x_i, y_i):
    """
    Reads the B02 BAND (10m) from all available SAFE dates
    and returns a np.array of shape (num_dates, 366, 366).

    If no B02 file exists, returns None.
    """
    band_name = "B02"
    b02_data_list = []

    for (dt_obj, safe_path) in safe_list:
        jp2_path = load_band_path(safe_path, band_name)
        if jp2_path is None:
            continue
        patch_data = read_patch_bands(jp2_path, x_i, y_i, band_name)
        if patch_data is None:
            continue
        b02_data_list.append(patch_data)

    if not b02_data_list:
        return None

    return np.stack(b02_data_list, axis=0)  # (t, 366, 366)



#############################
# MAIN FUNCTION
#############################

def main():
    args = parse_args()

    base_dir = Path(args.base_dir)

    for year in args.years:
        for tile_name in args.tiles:

            sentinel_dir = base_dir / "Imagenes Sentinel" / year / f"Tesela {tile_name}"
            parcels_raster = base_dir / "Parcelas CyL" / f"parcels{year}.tif"
            labels_raster  = base_dir / "Mapa ITACYL cultivos" / f"labels{year}.tif"

            out_dir = base_dir / year / tile_name
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_list = find_safe_dirs(sentinel_dir)
            if not safe_list:
                print(f"No .SAFE subdirectories found in {sentinel_dir}")
                continue

            ref_10m_path = None
            for (dt_obj, safe_path) in safe_list:
                p = load_band_path(safe_path, "B02")
                if p is not None:
                    ref_10m_path = p
                    break

            if not ref_10m_path:
                print(f"No .SAFE folder with B02 band found in {sentinel_dir}")
                continue
            total_patches = NX_PATCHES * NY_PATCHES
            print(f"Processing year={year}, tile={tile_name}...")

            for x_i in range(NX_PATCHES):
                for y_i in range(NY_PATCHES):
                    patch_index = x_i * NY_PATCHES + y_i + 1
                    progress = (patch_index / total_patches) * 100

                    # 1) Read parcels and labels to see if they are all zeros.
                    parcels_data = read_patch_same_area(ref_10m_path, x_i, y_i, "B02", str(parcels_raster))
                    labels_data  = read_patch_same_area(ref_10m_path, x_i, y_i, "B02", str(labels_raster))

                    # If both are 0 => skip
                    if (parcels_data.sum() == 0) and (labels_data.sum() == 0):
                        print(f"  -> Patch [{x_i},{y_i}] with no info (outside CyL). Skipped. Progress: {progress:.1f}%")
                        continue

                    # 2) Verify coverage in B02.
                    b02_stack = read_time_series_of_patch_B02(safe_list, x_i, y_i)
                    if b02_stack is None:
                        print(f"  -> Patch [{x_i},{y_i}] with no B02 data. Skipped. Progress: {progress:.1f}%")
                        continue

                    num_total_px = b02_stack.size
                    num_zeros = np.count_nonzero(b02_stack == 0)
                    frac_zeros = num_zeros / num_total_px

                    if frac_zeros > MAX_ZERO_PERCENT_B02:
                        print(f"  -> Patch [{x_i},{y_i}] with {frac_zeros*100:.1f}% zeros in B02. Skipped. Progress: {progress:.1f}%")
                        continue

                    # 3) If we pass the filters => create the .nc file
                    patch_name = f"patch_{x_i:02d}_{y_i:02d}"
                    patch_full_name = f"{year}_{tile_name}_{patch_name}"
                    nc_filename = f"{patch_full_name}.nc"
                    nc_filepath = out_dir / nc_filename

                    root_grp = netCDF4.Dataset(nc_filepath, 'w', format='NETCDF4')
                    # Global attributes
                    root_grp.title = "S4A Patch Dataset Castilla y León"
                    root_grp.authors = (
                        "Rodrigo Pascual-García (rodrigopg@ubu.es), "
                        "Pedro Latorre-Carmona, Jose Francisco Díez-Pastor / Universidad de Burgos"
                    )
                    root_grp.patch_full_name = patch_full_name
                    root_grp.patch_year = year
                    root_grp.patch_name = patch_name
                    root_grp.patch_country_code = COUNTRY_CODE
                    root_grp.patch_tile = tile_name
                    root_grp.creation_date = datetime.now().strftime('%d %b %Y')
                    root_grp.version = "21.03"

                    coords_10m_y = None
                    coords_10m_x = None

                    # 4) Store ALL bands (all dates)
                    for band_name, (subfolder, res_m) in BANDS_INFO.items():
                        band_time_data = []
                        band_time_stamps = []
                        raster_used = None

                        # Iterate over all dates
                        for (dt_obj, safe_path) in safe_list:
                            jp2_path = load_band_path(safe_path, band_name)
                            if jp2_path is None:
                                continue
                            patch_data = read_patch_bands(jp2_path, x_i, y_i, band_name)
                            if patch_data is None:
                                continue

                            band_time_data.append(patch_data)
                            band_time_stamps.append(time_to_datetime64(dt_obj))
                            if not raster_used:
                                raster_used = jp2_path

                        if len(band_time_data) == 0:
                            continue

                        t_len = len(band_time_data)
                        bh, bw = BAND_DIMENSIONS[band_name]

                        band_group = root_grp.createGroup(band_name)

                        band_scalar = band_group.createVariable('band', 'i8', ())
                        band_scalar[...] = 1
                        band_scalar.setncattr('standard_name', 'band_index')
                        band_scalar.setncattr('_IsNetcdf4Coordinate', 'true')

                        band_group.createDimension('time', t_len)
                        band_group.createDimension('y', bh)
                        band_group.createDimension('x', bw)

                        time_var = band_group.createVariable('time', 'f8', ('time',))
                        y_var = band_group.createVariable('y', 'f8', ('y',))
                        x_var = band_group.createVariable('x', 'f8', ('x',))

                        b_var = band_group.createVariable(
                            band_name, 'u2', ('time', 'y', 'x'), zlib=True
                        )

                        time_var.units = "days since 1970-01-01 00:00:00"
                        time_var.calendar = "proleptic_gregorian"

                        if raster_used:
                            y_vals, x_vals = compute_coords_bands(raster_used, x_i, y_i, band_name)
                        else:
                            y_vals = np.arange(bh)
                            x_vals = np.arange(bw)

                        y_var[:] = y_vals
                        x_var[:] = x_vals

                        def conv_dt64(dt64):
                            pydt = dt64_to_py_datetime(dt64)
                            return netCDF4.date2num(pydt, units=time_var.units, calendar=time_var.calendar)

                        time_arr = np.array([conv_dt64(d) for d in band_time_stamps], dtype=np.float64)
                        time_var[:] = time_arr

                        data_stacked = np.array(band_time_data, dtype=np.uint16)
                        b_var[:] = data_stacked

                        b_var.setncattr('coordinates', 'band')
                        b_var.transform = "..."
                        b_var.res = f"[{res_m}. {res_m}.]"
                        b_var.crs = "+init=epsg:32630"
                        b_var.is_tiled = "0"
                        b_var.nodatavals = "nan"
                        b_var.scales = "1.0"
                        b_var.offsets = "0.0"
                        b_var.AREA_OR_POINT = "Area"

                        if res_m == 10 and coords_10m_y is None and coords_10m_x is None:
                            coords_10m_y = y_vals
                            coords_10m_x = x_vals

                    # 5) Add labels
                    labels_grp = root_grp.createGroup('labels')
                    band_scalar = labels_grp.createVariable('band', 'i8', ())
                    band_scalar[...] = 1
                    band_scalar.setncattr('standard_name', 'band_index')
                    band_scalar.setncattr('_IsNetcdf4Coordinate', 'true')

                    labels_grp.createDimension('y', 366)
                    labels_grp.createDimension('x', 366)

                    y_var = labels_grp.createVariable('y', 'f8', ('y',))
                    x_var = labels_grp.createVariable('x', 'f8', ('x',))

                    if coords_10m_y is not None and coords_10m_x is not None:
                        y_var[:] = coords_10m_y
                        x_var[:] = coords_10m_x
                    else:
                        y_var[:] = np.arange(366)
                        x_var[:] = np.arange(366)

                    labels_var = labels_grp.createVariable('labels', 'u4', ('y', 'x'), zlib=True)
                    labels_var[:] = labels_data
                    labels_var.setncattr('coordinates', 'band')
                    labels_var.crs = "+init=epsg:32630"
                    labels_var.transform = "..."
                    labels_var.res = "[10. 10.]"
                    labels_var.nodatavals = "nan"

                    # 6) Add parcels
                    parcels_grp = root_grp.createGroup('parcels')
                    band_scalar = parcels_grp.createVariable('band', 'i8', ())
                    band_scalar[...] = 1
                    band_scalar.setncattr('standard_name', 'band_index')
                    band_scalar.setncattr('_IsNetcdf4Coordinate', 'true')

                    parcels_grp.createDimension('y', 366)
                    parcels_grp.createDimension('x', 366)

                    y_var = parcels_grp.createVariable('y', 'f8', ('y',))
                    x_var = parcels_grp.createVariable('x', 'f8', ('x',))

                    if coords_10m_y is not None and coords_10m_x is not None:
                        y_var[:] = coords_10m_y
                        x_var[:] = coords_10m_x
                    else:
                        y_var[:] = np.arange(366)
                        x_var[:] = np.arange(366)

                    parcels_var = parcels_grp.createVariable('parcels', 'u4', ('y', 'x'), zlib=True)
                    parcels_var[:] = parcels_data
                    parcels_var.setncattr('coordinates', 'band')
                    parcels_var.crs = "+init=epsg:32630"
                    parcels_var.transform = "..."
                    parcels_var.res = "[10. 10.]"
                    parcels_var.nodatavals = "nan"

                    # Close the dataset
                    root_grp.close()

                    # Show progress
                    print(f"  -> Created patch [{x_i},{y_i}] => {nc_filename}. "
                          f"Progress: {progress:.1f}%")

            print(f"Finished year={year}, tile={tile_name}.\n")


if __name__ == "__main__":
    main()

