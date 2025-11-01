# Análisis de Imágenes Térmicas para Predicción de Desarrollo Radicular

Este proyecto analiza imágenes térmicas y RGB de plantas para predecir su desarrollo radicular usando Machine Learning.

---

## ¿Qué hace este proyecto?

1. **Carga imágenes** (RGB, térmicas y máscaras de segmentación)
2. **Extrae características** de temperatura de hojas y tallos
3. **Analiza los datos** con PCA y correlaciones
4. **Predice** el desarrollo radicular con Regresión Logística

---

## Requisitos previos

- Python 3.11 o superior
- Miniconda instalado

---

## Instalación

### 1. Descargar el proyecto
Descarga o clona este repositorio en tu computadora.

### 2. Crear el entorno
Abre la terminal y ejecuta:
```bash
conda create -n mi_proyecto_env python=3.11
conda activate mi_proyecto_env
```

### 3. Instalar librerías
```bash
conda install -c conda-forge mahotas
pip install -r requirements.txt
```

---

## Preparar los datos

⚠️ **Los datos NO están incluidos en este repositorio.**

Para ejecutar el análisis:

1. Crea una carpeta llamada `datos/` en la carpeta del proyecto
2. Coloca dentro estos archivos:
   - `rooting_data.zip` (imágenes)
   - `rooting_data.csv` (información de las imágenes)

Tu estructura debe quedar así:
```
Romina_proyecto/
├── funciones_romina.py
├── analisis_romina.ipynb
├── requirements.txt
├── README.md
└── datos/                      ← Crea esta carpeta
    ├── rooting_data.zip        ← Pon tus datos aquí
    └── rooting_data.csv        ← Pon tus datos aquí
```

---

## Cómo ejecutar

### 1. Abrir VS Code
Abre VS Code en la carpeta del proyecto.

### 2. Abrir el notebook
Abre el archivo `analisis_romina.ipynb`

### 3. Activar el entorno
En la terminal de VS Code:
```bash
conda activate mi_proyecto_env
```

### 4. Ejecutar
Ejecuta las celdas del notebook en orden de arriba hacia abajo.

---

## Archivos del proyecto

- **`funciones_romina.py`**: Todas las funciones para procesar imágenes y extraer características
- **`analisis_romina.ipynb`**: Notebook principal con el análisis completo
- **`requirements.txt`**: Lista de librerías necesarias

---

## ¿Qué resultados obtengo?

El análisis genera:
- ✅ Características de temperatura extraídas de cada planta
- ✅ Gráficos de análisis exploratorio (PCA, correlaciones)
- ✅ Modelo de clasificación entrenado
- ✅ Métricas de desempeño (precisión, recall, F1-score)
- ✅ Matriz de confusión

---

## Notas técnicas

- Las imágenes térmicas están codificadas (0-255) y se decodifican automáticamente
- Se usan First Order Statistics (FOS) para extraer características
- El modelo clasifica en 2 categorías: desarrollo bajo (0) vs alto (1)

- Basado en umbral: root_scale < 3 = Clase 0, root_scale ≥ 3 = Clase 1
