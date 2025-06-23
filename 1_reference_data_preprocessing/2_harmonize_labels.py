import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from collections import Counter

# 1) READ THE ORIGINAL CSV
df_original = pd.read_csv("crop_id_mapping_SIGPAC.csv", header=None, names=["label", "code"])

# 2) EQUIVALENCE DICTIONARY: old_label -> (new_label, new_code)
equivalencias = {
    "PASTOS PERMANENTES DE 5 O MAS AÑOS": ("PASTOS PERMANENTES DE 5 O MÁS AÑOS", 255),
    "ALBARICOQUEROS": ("ALBARICOQUERO", 305),
    "MELOCOTONEROS": ("MELOCOTONERO", 277),
    "VIÑA- FRUTAL": ("VIÑA - FRUTAL", 303),
    "NECTARINOS": ("NECTARINO", 299),
    "FRAMBUESAS": ("FRAMBUESA", 294),
    "GUINDILLAS": ("GUINDILLA", 338),
    "ESPÁRRAGOS": ("ESPÁRRAGO", 275),
    "ALMENDROS": ("ALMENDRO", 256),
    "CAQUI O PALOSANTO": ("CAQUI o PALOSANTO", 302),
    "GARBANZOS": ("GARBANZO", 247),
    "COLES DE BRUSELAS": ("COL DE BRUSELAS", 329),
    "MEZCLA ALGARROBAS - AVENA": ("MEZCLA ALGARROBA-AVENA", 300),
    "TITARROS": ("TITARRO", 268),
    "CASTAÑOS": ("CASTAÑO", 279),
    "LENTEJAS": ("LENTEJA", 253),
    "OLIVAR-FRUTAL": ("OLIVAR - FRUTAL", 270),
    "PUERROS": ("PUERRO", 269),
    "VIVEROS": ("VIVERO", 298),
    "ALUBIAS": ("ALUBIA", 280),
    "CHOPOS": ("CHOPO", 261),
    "FRESAS": ("FRESA", 274),
    "TRANQUILLON": ("TRANQUILLÓN", 316),
    "EQUINÁCEA": ("EQUINACEA", 333),
    "HABAS": ("HABA", 281),
    "CHIRIVIA": ("CHIRIVÍA", 288),
    "LAVANDIN": ("LAVANDÍN", 266),
    "ARANDANO": ("ARÁNDANO", 292),
    "HIPERICO": ("HIPÉRICO", 332),
    "ALBERJON": ("ALBERJÓN", 311),
    "ALFORFON": ("ALFORFÓN", 286),
    "CARTAMO": ("CÁRTAMO", 265),
    "OREGANO": ("ORÉGANO", 330),
    "LUPULO": ("LÚPULO", 312),
    "TREBOL": ("TRÉBOL", 276),
    "NOGALES": ("NOGAL", 263),
    "PERALES": ("PERAL", 289),
    "SANDIA": ("SANDÍA", 278),
    "CAÑAMO.": ("CÁÑAMO", 324),
    "MEZCLA VEZA-TRITICALE": ("MEZCLA TRITICALE-VEZA", 282),
    "MAIZ": ("MAÍZ", 248),
    "VIÑEDO VINIFICACION": ("VIÑA", 249),
    "MEZCLA VEZA - AVENA": ("MEZCLA AVENA-VEZA", 271),
    "OLIVAR": ("OLIVO", 251)   
}


# A) UNIFY LABELS AND CODES IN DATAFRAME, BEFORE RE-INDEXING
def actualizar_etiqueta(row):
    """Applies the equivalence (old_label -> new_label, new_code)."""
    old_label = row["label"]
    if old_label in equivalencias:
        new_label, new_code = equivalencias[old_label]
        row["label"] = new_label
        row["code"] = new_code
    return row

df_unificado = df_original.apply(actualizar_etiqueta, axis=1)
df_unificado.drop_duplicates(inplace=True)


# B) RE-INDEX CODES WITHOUT GAPS
df_unificado.sort_values(by="label", inplace=True)

etiquetas_unicas = df_unificado["label"].unique()

nueva_numeracion = {etq: i for i, etq in enumerate(etiquetas_unicas, start=1)}

df_unificado["code"] = df_unificado["label"].map(nueva_numeracion)

df_unificado.sort_values(by="code", inplace=True)

df_unificado.to_csv("crop_id_mapping_SIGPAC_unif.csv", index=False, header=False)
print("'crop_id_mapping_SIGPAC_unif.csv' has been created with unified labels and re-indexed codes.\n")


# C) UPDATE THE RASTERS
new_codes_by_label = dict(zip(df_unificado["label"], df_unificado["code"]))

code_map = {}
for idx, row in df_original.iterrows():
    old_label = row["label"]
    old_code = row["code"]
    
    if old_label in equivalencias:
        new_label, _ = equivalencias[old_label]
        if new_label in new_codes_by_label:
            code_map[old_code] = new_codes_by_label[new_label]
        else:
            code_map[old_code] = old_code
    else:
        if old_label in new_codes_by_label:
            code_map[old_code] = new_codes_by_label[old_label]
        else:
            code_map[old_code] = old_code

def safe_get(x):
    """Returns the mapped code or, if it doesn't exist, keeps the original value x."""
    return code_map.get(x, x)

years = [2020, 2021, 2022, 2023, 2024]
for y in years:
    in_raster = f"labels{y}.tif"
    out_raster = f"labels{y}_unif.tif"
    
    if not os.path.isfile(in_raster):
        print(f"{in_raster} not found, skipping.")
        continue
    
    print(f"Updating codes in {in_raster} => {out_raster} ...")
    with rasterio.open(in_raster) as src:
        profile = src.profile.copy()
        with rasterio.open(out_raster, "w", **profile) as dst:
            for idx, window in src.block_windows(1):
                arr = src.read(1, window=window)
                arr_new = np.vectorize(safe_get)(arr)
                dst.write(arr_new, 1, window=window)
    print(f"Created => {out_raster}\n")

print("Process complete! Labels have been unified, codes re-indexed, and the .tif files have been updated.")