import sys

# from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import QApplication
from Stim_P24.Stim_Interface import StimInterfaceWidget
from Stim_P24.Stim_parameter import StimulatorSetUp
from Received_data_force import DataReceiver
import logging

# Utilisez QThread pour exécuter DataReceiver en parallèle
from PyQt5.QtCore import QThread


# Configure le logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    # Définition des paramètres du serveur
    server_ip = "192.168.0.1"  #   # "192.168.0.1" 127.0.0.1# Adresse IP du serveur
    server_port = 7  # 7  # 50000 Port à utiliser

    # Créez une application PyQt5
    app = QApplication(sys.argv)

    # Créez une instance du widget de visualisation
    visualization_widget = StimInterfaceWidget()
    visualization_widget.show()
    stimulator_configuration = StimulatorSetUp()

    # Créez une instance de DataReceiver
    data_receiver = DataReceiver(server_ip, server_port, visualization_widget)

    class DataThread(QThread):

        def __init__(self, receiver):
            super().__init__()
            self.receiver = receiver

        def run(self):
            self.receiver.start_receiving()

    data_thread = DataThread(data_receiver)
    data_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
