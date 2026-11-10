# 🌸 FloraVision AI · Clasificación Fina de Flores (Oxford 102)
> **Proyecto Final · Deep Learning · Universidad de La Sabana**

Aplicación web monolítica (MVP) para la identificación automática de especies florales a partir de fotografías, basada en el mejor modelo convolucional evaluado en el cuaderno de experimentación `Proyecto_Flores_Oxford102_v2.ipynb`.

---

## 📑 Tabla de Contenidos
1. [Descripción y Justificación del Proyecto](#-descripción-y-justificación-del-proyecto)
2. [Dataset: Oxford 102 Category Flower](#-dataset-oxford-102-category-flower)
3. [Modelos Evaluados en el Cuaderno (`ipynb`)](#-modelos-evaluados-en-el-cuaderno-ipynb)
   - [Modelo A: CNN Residual con Squeeze-and-Excitation](#modelo-a-cnn-residual-con-squeeze-and-excitation)
   - [Modelo B: Transfer Learning con ResNet-50 (Ganador)](#modelo-b-transfer-learning-con-resnet-50-ganador)
4. [Resultados y Comparativa de Métricas](#-resultados-y-comparativa-de-métricas)
5. [Arquitectura del Monolito (App Web)](#-arquitectura-del-monolito-app-web)
6. [Preprocesamiento e Inferencia](#-preprocesamiento-e-inferencia)
7. [Instalación y Ejecución en Local](#-instalación-y-ejecución-en-local)
8. [Despliegue en Render.com](#-despliegue-en-rendercom)
9. [Estructura del Repositorio](#-estructura-del-repositorio)

---

## 🌿 Descripción y Justificación del Proyecto

Identificar automáticamente la especie de una flor entre **102 categorías botánicas** a partir de una imagen real.

### ¿Por qué es relevante?
- **Floricultura y Agroindustria:** Colombia es uno de los mayores exportadores mundiales de flor cortada. La visión computacional automatiza el control de calidad, selección poscosecha y monitoreo de inventarios sin intervención destructiva.
- **Reto *Fine-Grained* (Clasificación de Grano Fino):** A diferencia de problemas con clases contrastantes (ej. perro vs. gato), las flores comparten patrones cromáticos y texturas similares con alta variabilidad intra-clase y sutiles diferencias morfológicas inter-clase.
- **Ciencia Ciudadana y Biodiversidad:** Soporte a iniciativas de conservación ambiental e inventario ecológico inspiradas en plataformas como *Pl@ntNet* o *iNaturalist*.

---

## 📊 Dataset: Oxford 102 Category Flower

- **Autoría:** Nilsback & Zisserman (2008), Universidad de Oxford.
- **Total de imágenes:** 8.189 fotografías a color (RGB).
- **Número de clases:** 102 especies de flores comunes del Reino Unido y orquídeas exóticas.
- **Distribución:** Entre 40 y 258 imágenes por especie (desbalance representativo de condiciones biológicas reales).
- **Limpieza de datos:** Se descartaron imágenes duplicadas exactas vía hash MD5 y se aplicó verificación de integridad antes del particionado estratificado.

---

## 🧠 Modelos Evaluados en el Cuaderno (`ipynb`)

En el cuaderno de experimentación se diseñaron, entrenaron y contrastaron dos enfoques arquitecturales:

### Modelo A: CNN Residual con Squeeze-and-Excitation
- **Enfoque:** Red convolucional entrenada **desde cero** (*from scratch*).
- **Topología:**
  - *Stem:* Convolución inicial 7×7 con stride 2, normalización por lotes (`BatchNorm2d`), activación `ReLU` y `MaxPool2d`.
  - *Etapas residuales:* 4 etapas progresivas (64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512 canales) con conexiones *skip* (residuales) y módulos de atención por canales **Squeeze-and-Excitation (SE-Blocks)** para recalibrar mapas de características informativos.
  - *Clasificador:* `AdaptiveAvgPool2d(1)`, `Flatten`, `Dropout(0.4)` y capa lineal densa de 512 $\rightarrow$ 102.
- **Entrenamiento:** Optimizador AdamW, programación de tasa de aprendizaje con decaimiento coseno, pérdida con *Label Smoothing* (0.1) y balanceo mediante `WeightedRandomSampler`.

### Modelo B: Transfer Learning con ResNet-50 (Ganador)
- **Enfoque:** Aprendizaje por transferencia adaptando una red profunda preentrenada en ImageNet-1K (v2).
- **Estrategia en 2 fases:**
  1. **Fase 1 (Backbone congelado):** Se congela el extractor de características y se entrena únicamente la cabeza clasificadora personalizada (`Dropout(0.3)` + `Linear(2048, 102)`) mediante búsqueda en rejilla (*grid search*) de hiperparámetros óptimos ($lr$, $dropout$, $weight\_decay$).
  2. **Fase 2 (Fine-Tuning diferencial):** Se descongela toda la red y se aplica ajuste fino con tasas de aprendizaje diferenciales:
     - Backbone: $1\times 10^{-4}$
     - Clasificador: $1\times 10^{-3}$
     - *Scheduler:* `OneCycleLR` con pico en 20% del ciclo.
- **Regularización y Aumento de Datos:** Rotaciones aleatorias, escalado afín, variación de color (*ColorJitter*), inversión horizontal y desenfoque gaussiano suave para simular fotos tomadas con teléfonos móviles.

---

## 📈 Resultados y Comparativa de Métricas

Evaluación realizada sobre el conjunto de prueba independiente (*test set*):

| Modelo | Arquitectura | Test Accuracy | F1-Score (Macro) | Top-5 Accuracy | Parámetros |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Modelo A** | CNN Propia (Residual + SE) | ~72.4% | ~0.710 | ~89.1% | ~11.2 M |
| **Modelo B** ⭐ | **Transfer Learning (ResNet-50)** | **97.6%** | **0.974** | **99.6%** | **~25.6 M** |

> **Conclusión del análisis:** El **Modelo B (ResNet-50)** superó holgadamente al modelo desde cero gracias al conocimiento previo de características visuales universales en capas tempranas, alcanzando un **99.6% en Top-5 Accuracy**, ideal para una aplicación de reconocimiento botánico.

---

## 🏛️ Arquitectura del Monolito (App Web)

La aplicación está construida como un **monolito ligero y autocontenido** optimizado para entornos de bajo consumo computacional (CPU) como el *Free Tier* de Render.com:

```text
┌────────────────────────────────────────────────────────┐
│                        Cliente                         │
│   (HTML5 + CSS3 Moderno + Vanilla JS Fetch API)        │
└──────────────────────────┬─────────────────────────────┘
                           │ POST /predict (multipart/form-data)
┌──────────────────────────▼─────────────────────────────┐
│                 Servidor Flask (Gunicorn)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pipeline de Preprocesamiento (PIL + NumPy)       │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │ Tensor (1, 3, 224, 224)  │
│  ┌──────────────────────────▼───────────────────────┐  │
│  │ Motor de Inferencia ONNX Runtime (CPU)           │  │
│  │ (Carga diferida y caché de sesión en memoria)    │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │ Logits -> Softmax        │
│  ┌──────────────────────────▼───────────────────────┐  │
│  │ Formateador de Respuesta JSON (Top-5 ordenado)   │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

- **Frontend dividido en 2 columnas:**
  - *Columna Izquierda:* Presentación comercial del proyecto, justificación técnica, métricas del cuaderno y comparativa de arquitecturas.
  - *Columna Derecha ("¡Pruébalo aquí!"):* Área de carga interactiva con soporte drag & drop, previsualización de imagen y gráfico de barras porcentuales con el Top-5 de predicciones.
- **Motor de Inferencia Dual:**
  - Utiliza preferentemente **ONNX Runtime** (`flower_model_b.onnx`) para tiempos de respuesta menores a 80 ms en CPU.
  - Posee mecanismo de contingencia para cargar checkpoints nativos de PyTorch (`model_b_best.pt`) si el runtime ONNX no estuviera disponible.

---

## ⚙️ Preprocesamiento e Inferencia

Para garantizar consistencia estricta con el cuaderno, el pipeline de preprocesamiento en producción replica la transformación de validación (`val_tf`):

1. **Decodificación de imagen:** Conversión forzada a espacio de color RGB.
2. **Redimensionado conservando aspecto:** Lado menor escalado a 256 px con interpolación bicúbica.
3. **Recorte central (*Center Crop*):** Extracción de la región central a 224×224 px.
4. **Normalización ImageNet:**
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
5. **Transposición dimensional:** $(H, W, C) \rightarrow (1, C, H, W)$.
6. **Softmax:**
   $$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^{102} e^{z_j}}$$

---

## 💻 Instalación y Ejecución en Local

### Prerrequisitos
- Python 3.10, 3.11 o 3.12 instalado.
- Git.

### 1. Clonar el repositorio y navegar a la carpeta
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd "proyecto final"
```

### 2. Crear y activar un entorno virtual
```bash
# En Windows:
python -m venv venv
.\venv\Scripts\activate

# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r app/requirements.txt
```

### 4. Asegurar el modelo entrenado
Verifica que el modelo exportado esté ubicado en la carpeta del modelo:
```text
app/model/flower_model_b.onnx
```
*(Si deseas reexportar el modelo desde el checkpoint PyTorch, ejecuta `python export_onnx.py`).*

### 5. Iniciar la aplicación
```bash
# Opción directa con Flask:
python app/app.py

# Opción usando Gunicorn (servidor de producción):
gunicorn --chdir app app:app --bind 127.0.0.1:5000
```

Abre tu navegador e ingresa a: **`http://localhost:5000`**

---

## 📂 Estructura del Repositorio

```text
proyecto final/
├── Proyecto_Flores_Oxford102_v2.ipynb    # Cuaderno de investigación y entrenamiento
├── export_onnx.py                        # Script de exportación de PyTorch a ONNX
├── render.yaml                           # Archivo de infraestructura para Render
├── .gitignore                            # Exclusiones de Git (cache, pesos pesados)
├── README.md                             # Documentación global del proyecto
│
└── app/                                  # Monolito web
    ├── app.py                            # API Flask, motor de inferencia y rutas
    ├── requirements.txt                  # Dependencias de producción
    ├── Procfile                          # Comando de arranque para Gunicorn en PaaS
    ├── README.md                         # Documentación interna de la app
    │
    ├── model/                            # Modelos de producción
    │   ├── flower_model_b.onnx           # Modelo ONNX compilado (~95 MB)
    │   └── model_b_best.pt               # Checkpoint de pesos PyTorch
    │
    └── templates/
        └── index.html                    # Frontend responsivo (Showcase + Playground)
```
