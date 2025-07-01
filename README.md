<!-- --------------------------------------------------------------------- -->
<!--         S4A-CyL – Sentinel-2 for Agriculture         -->
<!-- --------------------------------------------------------------------- -->
<p align="center">
  <img src="https://img.shields.io/github/last-commit/RodrigoPascual/S4A-CyL?style=flat-square">
  <img src="https://img.shields.io/github/languages/top/RodrigoPascual/S4A-CyL?style=flat-square">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square">
  <img src="https://img.shields.io/badge/data-CC%20BY%204.0-green?style=flat-square">
  <a href="https://hdl.handle.net/10259/10551"><img src="https://img.shields.io/badge/dataset-Riubu-orange?style=flat-square"></a>
</p>

<h1 align="center">S4A-CyL Dataset Creation</h1>
<p align="center"><em>Sentinel-2 time-series dataset with parcel-level crop labels for Castilla y León (Spain, 2020–2024)</em></p>

<!-- ===================================================================== -->
<!--                         Diagrama del Proyecto                         -->
<!-- ===================================================================== -->
<p align="center">
  <!-- 
    INSTRUCCIÓN: 
    1. Convierte tu PDF del diagrama a una imagen (p. ej., 'diagrama_s4a-cyl.png').
    2. Sube esa imagen a la carpeta raíz de tu repositorio.
    3. La imagen aparecerá automáticamente aquí. 
  -->
  <img src="diagrama_s4a-cyl.png" alt="Workflow for S4A-CyL Dataset Creation" width="850"/>
</p>

---

## Why this repository?

* **Explore quickly** – open any NetCDF patch in a couple of lines of Python or in the provided Jupyter notebook.  
* **Query with confidence** – utility functions to extract time-series, masks, parcel footprints, class names, etc.  
* **Extend if needed** – all preprocessing scripts are here, but they require the *official* SIGPAC shapefiles which are **not distributed in this repo** (see § Data prerequisites).

> **TL;DR** > If you only need to *use* the published dataset, clone the repo and jump straight to the notebook.  
> If you want to **re-generate** the raster layers or add future years, bring your own SIGPAC files and follow the full workflow.

---

## Table of Contents
- [Quick start](#quick-start)
- [Examples](#examples)
- [How to cite](#how-to-cite)
- [License](#license)

---

## Quick start

```bash
# 1. clone and enter
git clone [https://github.com/RodrigoPascual/S4A-CyL.git](https://github.com/RodrigoPascual/S4A-CyL.git)
cd S4A-CyL

# 2. create the conda env
conda env create -f environment.yml
conda activate s4acyl

# 3. download patches (≈ MB) from [https://hdl.handle.net/10259/10551](https://hdl.handle.net/10259/10551)

# 4. explore!
jupyter lab notebooks/patch_visualization.ipynb
```

## Data Structure

The S4A-CyL dataset is distributed as a collection of NetCDF (`.nc`) files. Each file, or "patch," is self-describing and contains multi-resolution time series data and the corresponding reference layers.

For a detailed breakdown of the hierarchical structure, including global attributes, dimensions, groups, and variables for a sample patch, please refer to the following file:

* **[NetCDF Patch Hierarchical Structure](NetCDF_patches_hierarchical_structure.txt)**


## Examples

| Patch · Band · Overlay | Imagen |
|-------------------------|--------|
| **2024_30TUK_patch_00_00** – Band B04 + Labels (grayscale) | <img src="Examples/patch_view_2024-30TUK-00_00_band-B04_t-0.png" width="520"/> |
| **2024_30TUM_patch_00_16** – RGB + Parcels (grayscale)  | <img src="Examples/patch_view_2024-30TUM-00_16_band-RGB Composite_t-0.png" width="520"/> |
| **2020_29TPF_patch_20_15** – RGB + Labels (vidris) + legend | <img src="Examples/patch_view_2020-29TPF-20_15_band-RGB Composite_t-3.png" width="520"/> |
| **2024_30TUM_patch_20_15** – RGB + Labels (color) + legend| <img src="Examples/patch_view_2024-30TUM-20_15_band-RGB Composite_t-0.png" width="520"/> |

