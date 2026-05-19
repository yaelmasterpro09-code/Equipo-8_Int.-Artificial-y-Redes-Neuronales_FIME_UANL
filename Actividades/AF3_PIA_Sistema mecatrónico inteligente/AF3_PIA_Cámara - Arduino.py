import cv2
import numpy as np
import tensorflow as tf
import serial
import time

MODEL_PATH = "modelo_basura.h5"
CLASSES_PATH = "clases.txt"

ARDUINO_PORT = "COM3"  # cambiar según en que port esté el Arduino
BAUD_RATE = 9600

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Clases cargadas:", class_names)

arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
time.sleep(2)

cap = cv2.VideoCapture(0)

ultima_clase = ""
ultimo_envio = 0
tiempo_espera = 3

def enviar_a_arduino(clase):
    if clase == "plastic":
        arduino.write(b'P')
        print("Arduino: plástico")
    elif clase == "cardboard":
        arduino.write(b'C')
        print("Arduino: cartón")
    elif clase == "metal":
        arduino.write(b'M')
        print("Arduino: metal")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo abrir la cámara.")
        break

    imagen = cv2.resize(frame, (224, 224))
    imagen = np.expand_dims(imagen, axis=0)

    prediccion = model.predict(imagen, verbose=0)
    indice = np.argmax(prediccion)
    confianza = prediccion[0][indice]
    clase = class_names[indice]

    texto = f"{clase}: {confianza:.2f}"
    cv2.putText(frame, texto, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.putText(frame, "ESPACIO = enviar | Q = salir",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.imshow("Clasificador de basura", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break

    if tecla == 32:  # ESPACIO
        print(f"Clase seleccionada: {clase} | Confianza: {confianza:.2f}")

        if confianza > 0.90:
            enviar_a_arduino(clase)
        else:
            print("Confianza baja, no se envió señal.")

cap.release()
arduino.close()
cv2.destroyAllWindows()