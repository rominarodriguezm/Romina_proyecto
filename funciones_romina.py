# =================== IMPORTS ===================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pyfeats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import zipfile

# =================== FIGURAS ===================
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """ Función para guardar figuras eficientemente """
    path = f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches="tight", transparent=True)

# =================== CARGA DE DATOS ===================
def decode_thermal_image(pil_image: Image.Image) -> np.ndarray:
    """ Decodifica una imagen PIL a valores de temperatura """
    min_temp = np.float32(pil_image.info["min_temp"])
    max_temp = np.float32(pil_image.info["max_temp"])
    encoded_array = np.array(pil_image, dtype=np.float32)
    thermal_array = (encoded_array / 255.0) * (max_temp - min_temp) + min_temp
    return thermal_array

def unzip_data(zip_path: str, extract_to: str = './'):
    """ Descomprime archivo ZIP """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Archivos extraídos en: {extract_to}")

def load_metadata_csv(csv_path: str, date_columns: list = None) -> pd.DataFrame:
    """ Carga CSV con metadatos """
    if date_columns:
        df = pd.read_csv(csv_path, parse_dates=date_columns)
    else:
        df = pd.read_csv(csv_path)
    return df

def check_missing_labels(df_metadata: pd.DataFrame, label_col: str, id_col: str) -> None:
    """ Verifica registros sin etiquetas """
    missing_count = len(df_metadata[df_metadata[label_col].isnull()][id_col].values.tolist())
    total_count = len(df_metadata)
    percentage = (missing_count / total_count) * 100
    print(f"Registros sin etiquetas: {missing_count} de {total_count} ({percentage:.5f}%)")

def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera reporte de valores nulos en un DataFrame
    
    Recibe
    -------
    df: pd.DataFrame
        DataFrame a analizar
    
    Devuelve
    --------
    pd.DataFrame
        DataFrame con reporte de nulos por columna conteniendo:
        - column: nombre de la columna
        - null_values: cantidad de valores nulos
        - null_percentage: porcentaje de valores nulos
    """
    report = pd.DataFrame({
        'column': df.columns,
        'null_values': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df)) * 100
    })
    return report

# =================== CARGA DE IMÁGENES ===================
def load_rgb_images(imgs_lst: list, df_metadata: pd.DataFrame, rgb_col: str, base_path: str = './') -> None:
    """ Carga imágenes RGB """
    for i in range(df_metadata.shape[0]):
        path = os.path.join(base_path, df_metadata[rgb_col][i])
        imgs_lst[0].append(np.array(Image.open(path)))
    print(f"✓ Cargadas {len(imgs_lst[0])} imágenes RGB")

def load_mask_images(imgs_lst: list, df_metadata: pd.DataFrame, mask_col: str, base_path: str = './', divisor: int = 120) -> list:
    """ Carga imágenes máscara """
    missing_files = []
    for i in range(df_metadata.shape[0]):
        path = os.path.join(base_path, df_metadata[mask_col][i])
        try:
            imgs_lst[1].append(np.array(Image.open(path)) // divisor)
        except FileNotFoundError:
            imgs_lst[1].append(np.nan)
            missing_files.append(path)
            print(f"Archivo no encontrado: {path}")
    print(f"✓ Cargadas {len(imgs_lst[1])} máscaras")
    if missing_files:
        print(f"⚠ Archivos faltantes: {len(missing_files)}")
    return missing_files

def load_thermal_images(imgs_lst: list, df_metadata: pd.DataFrame, thermal_col: str, decoder_func, base_path: str = './') -> None:
    """ Carga imágenes térmicas """
    for i in range(df_metadata.shape[0]):
        path = os.path.join(base_path, df_metadata[thermal_col][i])
        temp = Image.open(path)
        imgs_lst[2].append(decoder_func(temp))
    print(f"✓ Cargadas {len(imgs_lst[2])} imágenes térmicas")

def load_all_images(df_metadata: pd.DataFrame, rgb_col: str, mask_col: str,
                   thermal_col: str, decoder_func, base_path: str = './') -> tuple:
    """ Carga todos los tipos de imágenes """
    print("=" * 60)
    print("CARGANDO IMÁGENES")
    print("=" * 60)

    imgs_lst = [[], [], []]

    load_rgb_images(imgs_lst, df_metadata, rgb_col, base_path)
    missing_files = load_mask_images(imgs_lst, df_metadata, mask_col, base_path)
    load_thermal_images(imgs_lst, df_metadata, thermal_col, decoder_func, base_path)

    print("=" * 60)
    print(f"CARGA COMPLETADA - Total: {len(imgs_lst[0])} sets de imágenes")
    print("=" * 60)

    return imgs_lst, missing_files

def plot_sample_images(images_list: list, channel_names: list = None) -> None:
    """ Grafica aleatoriamente un set de imágenes """
    if channel_names is None:
        channel_names = ['Red', 'Green', 'Blue', 'Mask', 'Thermal']

    rnd_img = random.sample(range(len(images_list[0])), 1)[0]

    fig, axs = plt.subplots(1, 5, figsize=(15, 7))

    axs[0].imshow(images_list[0][rnd_img][:,:,0])
    axs[0].set_title(channel_names[0])
    axs[0].axis('off')

    axs[1].imshow(images_list[0][rnd_img][:,:,1])
    axs[1].set_title(channel_names[1])
    axs[1].axis('off')

    axs[2].imshow(images_list[0][rnd_img][:,:,2])
    axs[2].set_title(channel_names[2])
    axs[2].axis('off')

    axs[3].imshow(images_list[1][rnd_img])
    axs[3].set_title(channel_names[3])
    axs[3].axis('off')

    axs[4].imshow(images_list[2][rnd_img])
    axs[4].set_title(channel_names[4])
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Mostrando imagen #{rnd_img}")

# =================== EXTRACCIÓN FOS ===================
def es_mascara_valida(mask) -> bool:
    if isinstance(mask, (float, np.floating)):
        return not np.isnan(mask)
    return True

def obtener_roi(mask: np.ndarray, thermal: np.ndarray) -> tuple:
    combined_mask = (mask == 1) | (mask == 2)
    if not np.any(combined_mask):
        return None, None
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    margin_h = int(0.05 * (y_max - y_min))
    margin_w = int(0.05 * (x_max - x_min))
    y_min = max(0, y_min - margin_h)
    y_max = min(h-1, y_max + margin_h)
    x_min = max(0, x_min - margin_w)
    x_max = min(w-1, x_max + margin_w)
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
    roi_thermal = thermal[y_min:y_max+1, x_min:x_max+1] if thermal is not None else None
    return roi_mask, roi_thermal

def extraer_fos(imagen: np.ndarray, mascara: np.ndarray) -> dict:
    if not mascara.any():
        return {f'FOS_{nombre}': np.nan for nombre in [
            'Mean','Variance','Median','Mode','Skewness',
            'Kurtosis','Energy','Entropy','MinimalGrayLevel',
            'MaximalGrayLevel','CoefficientOfVariation',
            '10Percentile','25Percentile','75Percentile',
            '90Percentile','HistogramWidth']}
    features, names = pyfeats.fos(imagen, mascara)
    return {n: round(float(v),5) for n, v in zip(names, features)}

def procesar_region(imagen: np.ndarray, mascara: np.ndarray, prefijo: str, resultados: dict) -> None:
    fos = extraer_fos(imagen, mascara)
    for nombre, valor in fos.items():
        resultados[f'{prefijo}_{nombre}'] = valor

def procesar_banda_termica(thermal: np.ndarray, mask: np.ndarray, datos_toma: dict) -> None:
    procesar_region(thermal, mask == 1, "Termica_global_hojas", datos_toma)
    procesar_region(thermal, mask == 2, "Termica_global_tallos", datos_toma)

def generar_dataframe_estructurado(imgs_lst: list) -> pd.DataFrame:
    resultados_totales = []
    mascaras = imgs_lst[1]
    termicas = imgs_lst[2]

    for idx in tqdm(range(len(mascaras)), desc="Procesando tomas"):
        datos_toma = {'index': idx}
        try:
            mask = mascaras[idx]
            thermal = termicas[idx] if idx < len(termicas) else None
            if es_mascara_valida(mask) and not np.isnan(mask).all():
                roi_mask, roi_thermal = obtener_roi(mask, thermal)
                if roi_mask is not None and roi_thermal is not None:
                    procesar_banda_termica(roi_thermal, roi_mask, datos_toma)
        except Exception as e:
            print(f"Error en toma {idx}: {str(e)}")
        resultados_totales.append(datos_toma)

    df = pd.DataFrame(resultados_totales)
    return df.loc[:, ~df.columns.duplicated()].set_index('index')

# =================== PCA ===================
def estandarizar_datos(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def aplicar_pca(X_scaled: np.ndarray, n_componentes: int = 18) -> PCA:
    pca = PCA(n_components=n_componentes)
    pca.fit(X_scaled)
    print("Varianza explicada por cada componente:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    print(f"\nVarianza acumulada total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
    return pca

def graficar_varianza_pca(pca: PCA):
    componentes = range(1, len(pca.explained_variance_ratio_)+1)
    varianzas = pca.explained_variance_ratio_ * 100
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.figure(figsize=(12,6))
    plt.bar(componentes, varianzas, alpha=0.7, color='skyblue', label='Varianza individual (%)')
    plt.plot(componentes, varianza_acumulada, 'ro-', linewidth=2, label='Varianza acumulada (%)')
    plt.xlabel('Componente Principal')
    plt.ylabel('Porcentaje de Varianza Explicada (%)')
    plt.title('Varianza explicada por componentes principales', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
    plt.legend()
    plt.tight_layout()
    plt.show()

def obtener_loadings(pca: PCA, features: list) -> pd.DataFrame:
    loadings_matrix = pca.components_.T
    loadings_df = pd.DataFrame(loadings_matrix, index=features,
                               columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    return loadings_df

def graficar_top_variables(loadings_df: pd.DataFrame, data: pd.DataFrame, target: str, num_pc: int = 4, top_n: int = 5):
    variables_usadas = []
    todas_vars = []

    for i in range(num_pc):
        pc = f'PC{i+1}'
        pc_loadings = loadings_df[pc].abs().sort_values(ascending=False)
        pc_loadings = pc_loadings[~pc_loadings.index.isin(variables_usadas)]
        top_vars = pc_loadings.head(top_n)
        variables_usadas.extend(top_vars.index.tolist())
        todas_vars.extend(top_vars.index.tolist())

    clases = data[target].unique()
    medias = {clase: [data[data[target]==clase][feat].mean() for feat in todas_vars] for clase in clases}

    fig, ax = plt.subplots(figsize=(14,6))
    x_pos = np.arange(len(todas_vars))
    ancho = 0.35
    colores = ['red','green']
    for idx, clase in enumerate(clases):
        ax.bar(x_pos + (idx-0.5)*ancho, medias[clase], ancho, label=str(clase), color=colores[idx], alpha=0.7)

    etiquetas = [feat.replace("Termica_global_", "").replace("_FOS_", "_") for feat in todas_vars]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(etiquetas, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Temperatura promedio (°C)', fontsize=12, weight='bold')
    ax.set_title('Top variables de los componentes principales', fontsize=16, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=12)

    for i, _ in enumerate(todas_vars):
        for idx, clase in enumerate(clases):
            ax.text(i + (idx-0.5)*ancho, medias[clase][i]+0.05, f'{medias[clase][i]:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
