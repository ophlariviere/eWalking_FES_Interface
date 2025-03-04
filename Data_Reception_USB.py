import serial

# Modifier selon votre système :
PORT = "COM3"  # Windows -> "COM3" ou "COM4", Linux/Mac -> "/dev/ttyUSB0"
BAUDRATE = 115200  # Vitesse de communication (à vérifier avec la doc Cometa)

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"Connexion réussie sur {PORT}, lecture des données...")

    while True:
        line = ser.readline().strip()  # Lire une ligne de données
        if line:
            try:
                emg_values = [float(x) for x in line.decode('utf-8').split(',')]  # Convertir les données
                print(f"Données EMG reçues : {emg_values}")
            except ValueError:
                print("Erreur de conversion :", line)

except serial.SerialException as e:
    print(f"Erreur d'accès au port {PORT} : {e}")
except KeyboardInterrupt:
    print("\nArrêt du programme.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
