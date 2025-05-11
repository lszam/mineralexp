import os
import numpy as np
import glob
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import dask.array as da
from dask_ml.decomposition import PCA
import rioxarray
import rasterio
from rasterio.plot import show
from shapely.geometry import Polygon


# PARAMETROS
band_list = ["SR_B2","SR_B5","SR_B6","SR_B7"] # 👈 Bandas para hidroxilas
#band_list = ["SR_B2","SR_B4","SR_B5","SR_B6"] # 👈 Bandas para óxidos de ferro
pasta_landsat = "./python/landsat"
main_crs = "EPSG:4326" # 👈 CRS
k = 1 # f logistica - k menor, curva mais suave


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
            plt.savefig(os.path.join(pasta_landsat, f"pca_input_{banda}.png"), dpi=300, bbox_inches="tight")
            plt.show()

print("# ==================== PCA com Dask-ML ==================== #")

# AOI 
aoi = {
    "type": "Polygon",
    "coordinates": [[
        [-51.44, -8.05],          # SW
        [-48.72, -8.05],          # SE
        [-48.72, -4.93],          # NE
        [-51.44, -4.93],          # NW
        [-51.44, -8.05],          # fechar o polígono
    ]]
}
'''
#Area menor
aoi = {
    "type": "Polygon",
    "coordinates": [[
        [-52.50, -7.00],   # canto sudoeste 
        [-49.50, -7.00],   # canto sudeste
        [-49.50, -5.50],   # canto nordeste
        [-52.50, -5.50],   # canto noroeste
        [-52.50, -7.00]    # fechar o polígono
    ]]
}
'''

aoi_geom = Polygon(aoi["coordinates"][0])
aoi_gdf  = gpd.GeoDataFrame(geometry=[aoi_geom], crs=main_crs)

# 1. Ler e recortar cada mosaico em partes (chunks)
dataarrays = []
for banda in band_list:
    path = os.path.join(pasta_landsat, f"mosaico_{banda}.tif")
    da_banda = (
        rioxarray.open_rasterio(path, chunks=(1, 2048, 2048))
                 .squeeze("band") # remove dimensão band=1
                 .rio.clip(aoi_gdf.geometry, aoi_gdf.crs)  # recorte AOI
                 .astype("float32")
    )
    dataarrays.append(da_banda)

print("Empilhar bandas... shape das bandas = (bands, y, x)")
stack = da.stack([da.data for da in dataarrays], axis=0)

print("Preparar para PCA... shape = (pixels, bands)")
flat = stack.reshape((len(band_list), -1)).T
# rechunk para garantir que só haja um bloco na dimensão das bandas
flat = flat.rechunk({1: -1})

# Ajustar PCA “lazy”
print("Executar PCA...")
pca      = PCA(n_components=len(band_list), whiten=False)
pc_flat  = pca.fit_transform(flat)        # (pixels, components)

# Voltar ao cubo: shape = (components, y, x)
n_rows, n_cols = stack.shape[1:]
pc_cube = pc_flat.T.reshape((len(band_list), n_rows, n_cols))

'''# Visualizar
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
#plt.savefig(os.path.join(pasta_landsat, "pca_componentes_hx.png"), dpi=300, bbox_inches="tight") # HX
plt.savefig(os.path.join(pasta_landsat, "pca_componentes_fe.png"), dpi=300, bbox_inches="tight")
plt.show()'''

# Salvar cada componente em TIFF separado
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

print("# ==================== MAPAS “CROSTA” ==================== #")
# Montar o mapa de HIDROXILAS (H):
#      Em Loughlin 1991, para TM1,4,5,7 (aqui B2,B5,B6,B7) o PC4
#      destaca as Hidroxilas como pixels ESCUROS, então 
#      precisa inverter o sinal:
#      h_map = -pc4
# Montar o mapa de FERRO-ÓXIDO (F):
#      Em Loughlin 1991, para TM1,3,4,5 (aqui B2,B4,B5,B6) o PC4
#      destaca Ferro como pixels claros, então mantemos como está:
#      f_map = pc4

# Extrair PC4
pc4 = pc_cube[3].compute()    # PC4 como array 2D

# Puxar os loadings (autovetores) do modelo PCA
components = pca.components_
if hasattr(components, "compute"):
    components = components.compute()

pc4_loadings = components[3, :]  # vetor de 4 pesos, na ordem band_list
print("Loadings do PC4 (B2, B5, B6, B7):", pc4_loadings)

# Observar sinal de banda significativa
if pc4_loadings[band_list.index("SR_B7")] < 0:
    h_map = -pc4
    print("PC4 foi invertido (SR_B7 negativo).")
else:
    h_map = pc4
    print("PC4 não foi invertido (SR_B7 positivo).", )

#if pc4_loadings[band_list.index("SR_B4")] < 0:
#    f_map = -pc4
#    print("PC4 foi invertido (SR_B4 negativo).")
#else:
#    f_map = pc4
#    print("PC4 não foi invertido (SR_B4 positivo).", )

# Salvar e mostrar o mapa do alvo
out_h = os.path.join(pasta_landsat, "tcc_landsat_ mapa-crosta-hx.tif")
#out_h = os.path.join(pasta_landsat, "tcc_landsat_ mapa-crosta-fe.tif")

meta = metadados.copy()
try: 
    with rasterio.open(out_h, "w", **meta) as dst:
        dst.write(h_map.astype("float32"), 1)
        #dst.write(f_map.astype("float32"), 1)
    print("✅  Mapa salvo em", out_h)
except: 
    print("     Erro ao salvar o arquivo")

fig, ax = plt.subplots(1,1, figsize=(6,6))
im = ax.imshow(h_map, cmap="viridis")
#im = ax.imshow(f_map, cmap="viridis")
ax.set_title("Mapa Crosta – Hidroxilas (PC4)")
#ax.set_title("Mapa Crosta – Óx. Ferro (PC4)")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_mapa-crosta_hx.png"), dpi=300, bbox_inches="tight")
#plt.savefig(os.path.join(pasta_landsat, "pca_mapa-crosta_fe.png"), dpi=300, bbox_inches="tight")
plt.show()

# ==================== ANALISE DAS PCs ==================== #
print("# ==================== LOADINGS ==================== #")

# Extrair componentes (loadings) do PCA
components = pca.components_

# Para usar com Dask array
if hasattr(components, "compute"):
    components = components.compute()

print("\nAutovetores (loadings) de cada PC:")
for i, vec in enumerate(components):
    print(f"PC{i+1}:", dict(zip(band_list, np.round(vec, 3))))

#pc4 = components[3]   # vetor para PC4

# Calcular contribuição percentual de cada banda
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
fig.colorbar(im, ax=ax)#, fraction=0.046, pad=0.04)
ax.set_title("Percentual de contribuição das bandas em cada PC")
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_tabela_hx.png"), dpi=300, bbox_inches="tight")
#plt.savefig(os.path.join(pasta_landsat, "pca_tabela_fe.png"), dpi=300, bbox_inches="tight")
plt.show()



print("-------------------- Versão normalizada da PC4 --------------------")

def logistic_norm(arr, k=1.0, x0=None):
    """
    Aplica normalização logística ao array `arr`:
      S(x) = 1 / (1 + exp(-k*(x-x0)))
    - k  : slope parameter (default=1.0)
    - x0 : center (default=mean(arr))
    Retorna um array float em [0,1].
    """
    a = arr.astype("float32")
    if x0 is None:
        x0 = np.mean(a)
    return 1.0 / (1.0 + np.exp(-k * (a - x0)))


h_map_logist = logistic_norm(h_map, k=k)  
#f_map_logist = logistic_norm(f_map, k=k)  

fig, ax = plt.subplots(1,1, figsize=(6,6))
im = ax.imshow(h_map_logist, cmap="viridis")
#im = ax.imshow(f_map_logist, cmap="viridis")
ax.set_title("Mapa Crosta – Hidroxilas (PC4)")
#ax.set_title("Mapa Crosta – Óx. Ferro (PC4)")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(pasta_landsat, "pca_mapa-crosta-log_hx.png"), dpi=300, bbox_inches="tight")
#plt.savefig(os.path.join(pasta_landsat, f"pca_mapa-crosta_fe-logk{k}.png"), dpi=300, bbox_inches="tight")
plt.show()