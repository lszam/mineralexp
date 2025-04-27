import os
import numpy as np
import glob
import os
import rioxarray
import dask.array as da
from dask_ml.decomposition import PCA
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd

#========================================================= #

#band_list = ["SR_B2","SR_B5","SR_B6","SR_B7"] # 👈 Bandas para hidroxilas
band_list = ["SR_B2","SR_B4","SR_B5","SR_B6"] # 👈 Bandas para óxidos de ferro
pasta_landsat = "./python/landsat"
main_crs = "EPSG:4326" # 👈 CRS

# (OPCIONAL) Exibir mosaicos de entrada da PCA
for banda in band_list:
    # caminho do mosaico — ajuste se o padrão for diferente
    pattern = os.path.join(pasta_landsat, f"mosaico_{banda}.tif")
    for path in glob.glob(pattern):
        with rasterio.open(path) as src:
            fig, ax = plt.subplots(figsize=(10, 8))
            show(src, ax=ax, cmap="gray") 
            ax.set_title(f"Mosaico Landsat • {banda}")
            #ax.axis("off") 
            plt.show()


# ==================== PCA com Dask-ML ==================== #

# AOI 
aoi = {
    "type": "Polygon",
    "coordinates": [[
        [-51.44, -8.05],          # SW
        [-48.72, -8.05],          # SE
        [-48.72, -4.93],          # NE
        [-51.44, -4.93],          # NW
        [-51.44, -8.05],          # fecha
    ]]
}

aoi_geom = Polygon(aoi["coordinates"][0])
aoi_gdf  = gpd.GeoDataFrame(geometry=[aoi_geom], crs=main_crs)

# 1. Ler e recortar cada mosaico em partes (chunks)
dataarrays = []
for banda in band_list:
    path = os.path.join(pasta_landsat, f"mosaico_{banda}.tif")
    da_banda = (
        rioxarray.open_rasterio(path, chunks=(1, 2048, 2048))
                 .squeeze("band")                          # remove dimensão band=1
                 .rio.clip(aoi_gdf.geometry, aoi_gdf.crs)  # recorte AOI
                 .astype("float32")
    )
    dataarrays.append(da_banda)

# 2. Empilhar as bandas: shape = (bands, y, x)
stack = da.stack([da.data for da in dataarrays], axis=0)

# 3. Preparar para PCA: shape = (pixels, bands)
flat = stack.reshape((len(band_list), -1)).T
# Rechunk para que só haja um bloco na dimensão das bandas
flat = flat.rechunk({1: -1})

# 4. Ajustar PCA “lazy”
pca      = PCA(n_components=len(band_list), whiten=False)
pc_flat  = pca.fit_transform(flat)        # (pixels, components)

# 5. Voltar ao cubo: shape = (components, y, x)
n_rows, n_cols = stack.shape[1:]
pc_cube = pc_flat.T.reshape((len(band_list), n_rows, n_cols))

# 6. Visualizar
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx in range(len(band_list)):    # deve ser 4
    ax = axes[idx]
    pc = pc_cube[idx].compute()      # carrega só este componente
    im = ax.imshow(pc, cmap="gray")
    ax.set_title(f"PC {idx+1}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_componentes.png"), dpi=300, bbox_inches="tight")
plt.show()

# 7. Salvar cada componente em TIFF separado
modelo = os.path.join(pasta_landsat, f"mosaico_{band_list[0]}.tif")
with rasterio.open(modelo) as ref:
    metadados = ref.meta.copy()
metadados.update(count=1, dtype="float32")

for idx in range(len(band_list)):
    comp = pc_cube[idx].compute()
    out_path = os.path.join(pasta_landsat, f"PCA{idx+1}.tif")
    try:
        with rasterio.open(out_path, "w", **metadados) as dst:
            dst.write(comp, 1)
        print(f"✅  Salvo: {out_path}")
    except:
        print(f"Erro ao gravar arquivo {out_path}")
        continue

# ==================== MAPAS “CROSTA” (HIDROXILA e FERRO-ÓXIDO) ====================

# 5.b) Extrair o quarto componente (PC4) já calculado
#    pc_cube.shape == (4, n_rows, n_cols)
pc4 = pc_cube[3].compute()    # PC4 como array 2D

# 5.c) Puxar os loadings (autovetores) do modelo PCA
components = pca.components_
if hasattr(components, "compute"):
    components = components.compute()

pc4_loadings = components[3, :]    # vetor de 4 pesos, na ordem band_list
print("Loadings do PC4 (B2, B5, B6, B7):", pc4_loadings)

# 5.d) Montar o mapa de HIDROXILAS (H):
#      Em Loughlin 1991, para TM1,4,5,7 (aqui B2,B5,B6,B7) o PC4
#      destaca as Hidroxilas como pixels ESCUROS, então 
#      precisa inverter o sinal:
# h_map = -pc4

# 5.e) Montar o mapa de FERRO-ÓXIDO (F):
#      Em Loughlin 1991, para TM1,3,4,5 (aqui B2,B4,B5,B6) o PC4
#      destaca Ferro como pixels claros, então mantemos como está:
f_map = pc4
  

# 5.f) Salvar e mostrar o mapa do alvo
out_h = os.path.join(pasta_landsat, "tcc_landsat_crosta_FE.tif")
#out_h = os.path.join(pasta_landsat, "tcc_landsat_crosta_FE.tif")

meta = metadados.copy()
try: 
    with rasterio.open(out_h, "w", **meta) as dst:
        dst.write(f_map.astype("float32"), 1)
    print("✅  Mapa de Hidroxilas salvo em", out_h)
except: 
    print("     Erro ao salvar o arquivo")

fig, ax = plt.subplots(1,1, figsize=(6,6))
im = ax.imshow(f_map, cmap="magma")
ax.set_title("Crosta – Hidroxilas (PC4 com sinal invertido)")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_mapa-crosta_fe.png"), dpi=300, bbox_inches="tight")
plt.show()

# ==================== ANALISE DAS PCs ==================== #

# Extrair componentes (loadings) do PCA
components = pca.components_

# Para usar com Dask array
if hasattr(components, "compute"):
    components = components.compute()

# Calcular contribuição percentual de cada banda ──
loadings_sq = components**2
pct = loadings_sq / loadings_sq.sum(axis=1)[:, None] * 100

# Montar e exibir tabela 
df = pd.DataFrame(pct, index=[f"PC{i+1}" for i in range(pct.shape[0])],
    columns=band_list)
print("\n📊 Contribuição percentual de cada banda por componente:")
print(df.to_string(float_format="%.2f"))

# Imagem colorida dos valores
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(df.values, aspect='equal')  
ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.index)))
ax.set_xticklabels(df.columns, rotation=45, ha='right')
ax.set_yticklabels(df.index)
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.iloc[i, j] # anotar cada valor da contribuição 
        ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color='white' if val<50 else 'black')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Percentual de contribuição das bandas em cada PC")
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_tabela.png"), dpi=300, bbox_inches="tight")
plt.show()