"""
AF3 - Producto Integrador de Aprendizaje - Sistema Mecatrónico Inteligente

Datos del equipo:

- Edgar Yael Martínez Pérez- 2050426

- Edwin Magdaleno Domínguez 2177965

- Patricio Israel Gómez Martínez - 2062237 - N5

- Giann Luca Verdeja Briones - 2050452

Materia: Inteligencia Artificial y Redes Neuronales - Hora: V6 

"""

# ============================================================
# 1. CARGAR LIBRERÍAS
# ============================================================

# TensorFlow se utiliza para crear, entrenar y guardar el modelo de red neuronal.
import tensorflow as tf

# layers y models permiten construir la arquitectura del modelo por capas.
from tensorflow.keras import layers, models

# MobileNetV2 es una red neuronal convolucional preentrenada.
# Se utiliza como base para aplicar transferencia de aprendizaje.
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# 2. CARGAR DATOS
# ============================================================

# Tamaño al que se redimensionarán todas las imágenes.
# MobileNetV2 trabaja comúnmente con imágenes de 224x224 píxeles.
IMG_SIZE = (224, 224)

# Número de imágenes que se procesan por lote durante el entrenamiento.
BATCH_SIZE = 16

# Carpeta donde se encuentra el dataset organizado por clases.
DATASET_DIR = "dataset"

# Carga las imágenes de entrenamiento desde la carpeta dataset.
# Las etiquetas se asignan automáticamente según el nombre de cada carpeta.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,     # 20% de las imágenes se reserva para validación.
    subset="training",        # Se indica que este conjunto será de entrenamiento.
    seed=123,                 # Semilla para mantener la misma división de datos.
    image_size=IMG_SIZE,      # Redimensiona todas las imágenes a 224x224.
    batch_size=BATCH_SIZE     # Agrupa las imágenes en lotes de 16.
)

# Carga las imágenes de validación usando el mismo dataset.
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,     # Usa el mismo 20% reservado.
    subset="validation",      # Se indica que este conjunto será de validación.
    seed=123,                 # Misma semilla para que la división sea consistente.
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Obtiene los nombres de las clases detectadas a partir de las carpetas.
# Ejemplo: ['cardboard', 'metal', 'plastic']
class_names = train_ds.class_names

# Muestra las clases en consola para verificar que se cargaron correctamente.
print("Clases detectadas:", class_names)


# ============================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================

# AUTOTUNE permite que TensorFlow optimice automáticamente la carga de datos.
AUTOTUNE = tf.data.AUTOTUNE

# cache() almacena datos en memoria para acelerar el entrenamiento.
# shuffle(1000) mezcla las imágenes para evitar que el modelo aprenda por orden.
# prefetch() prepara el siguiente lote mientras el modelo entrena el actual.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# En validación no se mezclan los datos, solo se optimiza la carga.
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# 4. SELECCIÓN Y ENTRENAMIENTO DEL MODELO
# ============================================================

# Se carga MobileNetV2 como modelo base.
# input_shape indica que las imágenes son de 224x224 con 3 canales RGB.
# include_top=False elimina la capa clasificadora original de MobileNetV2.
# weights="imagenet" carga pesos ya entrenados con la base de datos ImageNet.
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Se congelan los pesos del modelo base para no modificarlos durante el entrenamiento.
# Esto permite usar MobileNetV2 como extractor de características.
base_model.trainable = False

# Se construye el modelo completo de manera secuencial.
model = models.Sequential([
    # Normaliza los valores de los píxeles de 0-255 al rango aproximado de -1 a 1.
    layers.Rescaling(1./127.5, offset=-1),

    # Modelo base MobileNetV2, encargado de extraer características visuales.
    base_model,

    # Reduce los mapas de características a un vector más pequeño.
    layers.GlobalAveragePooling2D(),

    # Apaga aleatoriamente el 30% de conexiones para reducir sobreajuste.
    layers.Dropout(0.3),

    # Capa final con una neurona por clase.
    # softmax convierte la salida en probabilidades.
    layers.Dense(len(class_names), activation="softmax")
])

# Se configura el modelo para el entrenamiento.
model.compile(
    optimizer="adam",                        # Optimizador Adam.
    loss="sparse_categorical_crossentropy",  # Función de pérdida para clasificación multiclase.
    metrics=["accuracy"]                     # Métrica utilizada para evaluar exactitud.
)

# Se entrena el modelo con el conjunto de entrenamiento y se valida con val_ds.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10                                # Número de épocas de entrenamiento.
)


# ============================================================
# 5. PRUEBA DEL MODELO
# ============================================================

# Evalúa el modelo usando el conjunto de validación.
loss, accuracy = model.evaluate(val_ds)

# Imprime la pérdida obtenida en validación.
print("Pérdida en validación:", loss)

# Imprime la exactitud obtenida en validación.
print("Exactitud en validación:", accuracy)

# Guarda el modelo entrenado en un archivo .h5.
# Este archivo será utilizado después por el programa de cámara.
model.save("modelo_basura.h5")

# Guarda los nombres de las clases en un archivo de texto.
# Esto permite que el código de cámara interprete correctamente la predicción.
with open("clases.txt", "w") as f:
    for clase in class_names:
        f.write(clase + "\n")

print("Modelo guardado como modelo_basura.h5")
print("Clases guardadas en clases.txt")