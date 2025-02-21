import serial
# Ouvrir la connexion série (remplacez COMx par le bon port sous Windows, ou /dev/ttyUSBx sous Linux/Mac)
ser = serial.Serial(port="COM4", baudrate=9600, timeout=1)

while True:
    line = ser.readline().decode('utf-8').strip()  # Lire une ligne de données
    if line:
        print(f"Données reçues : {line}")