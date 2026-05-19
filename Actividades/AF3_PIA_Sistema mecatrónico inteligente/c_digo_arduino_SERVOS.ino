#include <Servo.h>

Servo servoPlastico;
Servo servoCarton;
Servo servoMetal;

char dato;

const int pinPlastico = 9;
const int pinCarton = 10;
const int pinMetal = 11;

void setup() {
  Serial.begin(9600);

  servoPlastico.attach(pinPlastico);
  servoCarton.attach(pinCarton);
  servoMetal.attach(pinMetal);

  cerrarTodos();
}

void loop() {
  if (Serial.available() > 0) {
    dato = Serial.read();

    cerrarTodos();

    if (dato == 'P') {
      abrirContenedor(servoPlastico);
    }
    else if (dato == 'C') {
      abrirContenedor(servoCarton);
    }
    else if (dato == 'M') {
      abrirContenedor(servoMetal);
    }
  }
}

void abrirContenedor(Servo &servo) {
  // Abrir despacio de 0° a 90°
  for (int angulo = 0; angulo <= 90; angulo++) {
    servo.write(angulo);
    delay(5);
  }

  // Mantener abierto
  delay(4000);

  // Cerrar despacio de 90° a 0°
  for (int angulo = 90; angulo >= 0; angulo--) {
    servo.write(angulo);
    delay(5);
  }
}

void cerrarTodos() {
  servoPlastico.write(0);
  servoCarton.write(0);
  servoMetal.write(0);
}
