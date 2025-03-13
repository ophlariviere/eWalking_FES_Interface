import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging


class DataReceiver(QObject):
    def __init__(
            self,
            server_ip,
            server_port,
            visualization_widget,
            read_frequency=100,
            threshold=30,
    ):
        super().__init__()
        self.visualization_widget = visualization_widget
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcp_client = TcpClient(
            self.server_ip, self.server_port, read_frequency=read_frequency
        )


    def start_receiving(self):
        logging.info("Début de la réception des données...")
        while True:
            tic = time.time()
            for _ in range(3):  # Tentatives multiples
                try:
                    received_data = self.tcp_client.get_data_from_server()
                    break  # Si réussi, quittez la boucle
                except Exception as e:
                    logging.warning(f"Tentative échouée : {e}")
                    time.sleep(5)  # Attente avant la prochaine tentative
            else:
                logging.error("Impossible de se connecter après plusieurs tentatives.")
                continue

            if self.visualization_widget.dolookneedsendstim is True:
                self.check_stimulation(received_data)

            #self.process_data(received_data)

            loop_time = time.time() - tic
            real_time_to_sleep = max(0, 1 / self.read_frequency - loop_time)
            time.sleep(real_time_to_sleep)
