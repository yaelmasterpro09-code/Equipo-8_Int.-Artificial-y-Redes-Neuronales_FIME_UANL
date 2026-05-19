# ============================================================
# 1. CARGA DE LIBRERÍAS
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# 2. CARGA DE DATOS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_DIR = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)


# ============================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE
# Optimización automática del flujo de datos
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# 4. SELECCIÓN Y ENTRENAMIENTO DEL MODELO
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# ============================================================
# 5. PRUEBA DEL MODELO
# ============================================================

loss, accuracy = model.evaluate(val_ds)

print("Pérdida en validación:", loss)
print("Exactitud en validación:", accuracy)

model.save("modelo_basura.h5")

with open("clases.txt", "w") as f:
    for clase in class_names:
        f.write(clase + "\n")

print("Modelo guardado como modelo_basura.h5")
print("Clases guardadas en clases.txt")