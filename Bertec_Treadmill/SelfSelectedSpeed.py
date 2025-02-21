import socket
import numpy as np


def open_treadmill_comm():
    # Création de la connexion TCP/IP à localhost:4000
    t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32)  # Taille tampon de réception
    t.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64)  # Taille tampon d'envoi

    # Connexion à localhost sur le port 4000
    t.connect(('localhost', 4000))

    return t


def int16_to_bytes(numbers):
    # Vérifie que les nombres sont dans l'intervalle [-2^15, 2^15 - 1]
    numbers = np.array(numbers, dtype=int)
    n = len(numbers)

    bytes_array = np.zeros((n, 2), dtype=np.uint8)

    for i in range(n):
        if numbers[i] > (2 ** 15 - 1):
            print(f"Warning: number out of range for conversion. Saturating entry #{i + 1}")
            numbers[i] = 2 ** 15 - 1
        elif numbers[i] < (-2 ** 15):
            print(f"Warning: number out of range for conversion. Saturating entry #{i + 1}")
            numbers[i] = -2 ** 15

        if numbers[i] < 0:  # Nombres négatifs
            aux = 2 ** 15 + numbers[i]  # Calcul du complément à 2
            byte1 = 128  # Le bit de signe est stocké dans byte1 (MSB)
        else:  # Nombres positifs
            aux = numbers[i]
            byte1 = 0  # Le bit de signe est 0 (LSB)

        byte1 = byte1 + (aux // 2 ** 8)  # Premier octet : signe + MSB
        byte2 = aux - 2 ** 8 * (aux // 2 ** 8)  # Deuxième octet : LSB

        bytes_array[i, 0] = byte1
        bytes_array[i, 1] = byte2

    return bytes_array


if __name__ == "__main__":
    open_treadmill_comm()
