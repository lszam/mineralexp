from datetime import datetime
from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd
import pandas as pd
import numpy as np
#import re
import rasterio
from shapely.geometry import shape, Polygon
import matplotlib.pyplot as plt

import os
import requests
import sys

from rasterio.warp import calculate_default_transform, reproject#, Resampling
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt
import glob

from rasterio.enums import Resampling

from rasterio.transform import array_bounds
from affine import Affine

import subprocess
from shlex import split          # garante quebra correta da string em argumentos



#==============================================================#

band_list = ["SR_B2","SR_B5","SR_B6","SR_B7"] # 👈 BANDAS
pasta_landsat = "./python/landsat"
pasta_landsat_reproj = str(pasta_landsat + "_reproj")
reproj_crs = "EPSG:4326" # 👈 CRS

#==============================================================#


# ==================== Selecionar imagens ==================== #
# 1. Conectar ao catálogo
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# 2. Definir a área de interesse (AOI)
aoi = {
    "type": "Polygon",
    "coordinates": [[
        [-52.502096494512415, -7.005249818536389],   # canto sudoeste (xmin, ymin)
        [-49.50086144162274,  -7.005249818536389],   # canto sudeste (xmax, ymin)
        [-49.50086144162274,  -5.496190990780593],   # canto nordeste (xmax, ymax)
        [-52.502096494512415, -5.496190990780593],   # canto noroeste (xmin, ymax)
        [-52.502096494512415, -7.005249818536389]    # fechar o polígono
    ]]
}

# Salvar AOI também como GeoDataFrame
aoi_geom = Polygon(aoi['coordinates'][0])
aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")

# 3. Buscar imagens Landsat 8 (filtro nuvens e data)
print(" Buscar imagens no repositório...")
search = catalog.search(
    filter_lang="cql2-json",
    filter={
        "op": "and",
        "args": [
            {"op": "s_intersects", "args": [{"property": "geometry"}, aoi]},
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 1]}, # 👈 escolher % de nuvens
            {"op": "=", "args": [{"property": "collection"}, "landsat-8-c2-l2"]}, # 👈 escolher LANDSAT
            {"op": ">", "args": [{"property": "datetime"}, "2020-01-01T00:00:00Z"]},
            {"op": "<", "args": [{"property": "datetime"}, "2022-01-01T23:59:59Z"]}
        ]
    }
)

# Obter e ordenar por menor cobertura de nuvem
items = list(search.items())
items_sorted = sorted(items, key=lambda item: item.properties.get("eo:cloud_cover", 100))
# Selecionar as melhores
best_items = items_sorted[:100] # 👈 escolher número de imagens listadas
print(f"\n🟢 Existem {len(best_items)} imagens pré-selecionadas")

# Gerar GeoDataFrame - footprints imagens + cloudcover
geoms = [shape(item.geometry) for item in best_items]
images_gdf = gpd.GeoDataFrame({
    "cloud_cover": [item.properties["eo:cloud_cover"] for item in best_items],
    "datetime": [item.properties["datetime"] for item in best_items]
}, geometry=geoms, crs="EPSG:4326")

# Reprojetar para UTM 22S (projeção métrica em metros)
images_gdf_proj = images_gdf.to_crs("EPSG:31982")
# Calcular área no CRS projetado (em m²)
#images_gdf_proj["area"] = images_gdf_proj.geometry.area


# ==================== Identificar quais imagens manter ====================#
# escolher 1 cena por (path,row) — a de menor nuvem

pathrow_to_idx = {}  # (path,row) é uam par de índices em best_items

for idx, item in enumerate(best_items):
    key = (item.properties["landsat:wrs_path"],
           item.properties["landsat:wrs_row"])
    cc  = item.properties["eo:cloud_cover"]

    # Se ainda não tem entrada ou se achou cobertura menor, substitui
    if key not in pathrow_to_idx or cc < best_items[pathrow_to_idx[key]].properties["eo:cloud_cover"]:
        pathrow_to_idx[key] = idx

keep_idx = list(pathrow_to_idx.values())        # lista de índices inteiros

images_gdf_reduzido = images_gdf.iloc[keep_idx].copy()

print(f"▶︎ {len(keep_idx)} cenas mantidas (1 por path/row) de {len(images_gdf)}\n")
print(images_gdf_reduzido[["cloud_cover", "datetime", "geometry"]])


# ==================== BAIXAR BANDAS ==================== #

images_gdf_reduzido = images_gdf_reduzido.reset_index().rename(columns={'index':'orig_index'})

selected_items = [best_items[i] for i in images_gdf_reduzido['orig_index']]

# Pasta para salvar os arquivos
os.makedirs(pasta_landsat, exist_ok=True)

print(f"Baixar imagens em {pasta_landsat}")

for i, item in enumerate(selected_items, start=1):
    signed = pc.sign_item(item)
    for band_name in band_list:
        if band_name not in signed.assets:
            continue
        asset = signed.assets[band_name]
        out_path = str(pasta_landsat + f"\img{i}_{band_name}.tif")
        if not os.path.exists(out_path):
            print(f"  Baixando {band_name} da imagem {i}...")
            r = requests.get(asset.href, stream=True)
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

# REPROJETAR IMAGENS PARA UM MESMO CRS

print(f"🌐 Verificar CRS das imagens salvas... CRS a adotar: {reproj_crs}")

# Obter todos os .tif da pasta
img_paths = sorted(glob.glob(os.path.join(pasta_landsat, f"*.tif")))

# Ordenar com a função sort_key e exibir
images = sorted(img_paths)#, key=sort_key)
print("Arquivos:", images)

for img_path in images:
    # Abrir arquivo somente leitura
    with rasterio.open(img_path) as src:
        if src.crs == reproj_crs:
            print(f"{os.path.basename(img_path)} já está em {reproj_crs}")
            continue

        print(f"  Reprojetando {os.path.basename(img_path)} → {reproj_crs}...")
        old_crs       = src.crs
        old_transform = src.transform
        data          = src.read(1)

        transform, width, height = calculate_default_transform(
            old_crs, reproj_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': reproj_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    # Remover o TIFF antigo
    os.remove(img_path)

    # Reabrir no mesmo caminho para escrita
    with rasterio.open(img_path, 'w', **kwargs) as dst:
        try:
            reproject(
                source=data,
                destination=rasterio.band(dst, 1),
                src_transform=old_transform,
                src_crs=old_crs,
                dst_transform=transform,
                dst_crs=reproj_crs,
                resampling=Resampling.nearest
            )
        except:
            print(f"Erro ao reprojetar {os.path.basename(img_path)}")
            continue

        print(f"CRS reprojetado para {reproj_crs} em {img_path}")

# ==================== CORREÇÃO DO HISTOGRAMA ==================== #

for banda in band_list:
    print(f"\n🔧  Stretch banda {banda}")
    img_paths = sorted(glob.glob(os.path.join(pasta_landsat, f"img*_{banda}.tif")))
    if not img_paths:
        print("  Nenhum arquivo encontrado")
        continue

    # Acumular valores válidos
    all_vals = []
    for p in img_paths:
        with rasterio.open(p) as src:
            arr  = src.read(1).astype("float32")
            mask = (arr == src.nodata) | ~np.isfinite(arr) | (arr <= 0)
            valid = arr[~mask]
            if valid.size:
                all_vals.append(valid)

    if not all_vals:
        print("  Todas as cenas são NoData")
        continue

    p2, p98 = np.percentile(np.concatenate(all_vals), (2, 98))
    print(f"  p2={p2:.2f}  p98={p98:.2f}")

    # Aplicar o stretch em cada cena
    for p in img_paths:
        with rasterio.open(p) as src:
            arr  = src.read(1).astype("float32")
            mask = (arr == src.nodata) | ~np.isfinite(arr) | (arr <= 0)

            stretched = np.clip((arr - p2) / (p98 - p2), 0, 1)
            stretched[mask] = 0  # bordas = 0

            meta = src.meta.copy()
            meta.update(dtype="float32", nodata=0, count=1)

            out_path = p.replace(".tif", "_str.tif")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(stretched, 1)

            print("    → Arquivo salvo ", os.path.basename(out_path))


# ==================== MOSAICO (Rasterio) ==================== #
print("\n🧩  Criando mosaicos com Rasterio...")

nodata_val = 0

for banda in band_list:
    # Localizar TIFFs corrigidos da banda
    padrao = os.path.join(pasta_landsat, f"img*_{banda}_str.tif")
    img_paths = sorted(glob.glob(padrao))

    if not img_paths:
        print(f"⚠️ Nenhum arquivo encontrado para {banda}")
        continue

    print(f"Banda {banda}: {len(img_paths)} arquivos")

    # Abrir os rasters
    srcs = [rasterio.open(p) for p in img_paths]
    # Executar o merge usando o transform do primeiro raster como referência
    mosaic, out_transform = merge(srcs, nodata=nodata_val)
    # Preparar metadados com as dimensões do mosaico
    meta = srcs[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], 
                 "transform": out_transform, "nodata": nodata_val, "dtype": "float32"})

# Gravar o GeoTIFF final
mosaic_path = os.path.join(pasta_landsat, f"mosaico_{banda}.tif")
try: 
    with rasterio.open(mosaic_path, "w", **meta) as dst:
        dst.write(mosaic)
        meta = srcs[0].meta.copy()
        meta.update({ "height":  mosaic.shape[1], "width":   mosaic.shape[2],
                    "transform": out_transform })
        print("✅  Mosaico criado em", mosaic_path)
except:
    print("   ❌ Erro ao gravar arquivo de saída")

# Fechar arquivos de origem
for s in srcs:
    s.close()