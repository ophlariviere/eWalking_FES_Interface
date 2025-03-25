import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
from collections import deque
import numpy as np


class DataReceiver(QObject):
    def __init__(
        self,
        server_ip,
        server_port,
        visualization_widget,
        read_frequency=100,
    ):
        super().__init__()
        self.visualization_widget = visualization_widget
        self.server_ip = server_ip
        self.server_port = server_port
        self.read_frequency = read_frequency
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=self.read_frequency)
        logging.basicConfig(level=logging.INFO)
        self.sendStim = {1: False, 2: False}
        self.last_channels = []

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        buffer = deque(maxlen=10)
        while True:
            for _ in range(3):  # Multiple attempts
                try:
                    # Attempt to receive data from the server
                    received_data = self.tcp_client.get_data_from_server(
                        command=["footswitch_data"]
                    )

                    # Stim gestion
                    if received_data["footswitch_data"]:  # Ensure we have valid data before proceeding
                        footswitch_data_oneframe = received_data["footswitch_data"]
                        buffer.append(footswitch_data_oneframe)
                        if self.visualization_widget.dolookneedsendstim:
                            emg_num = self.visualization_widget.foot_emg
                            if emg_num:
                                footswitch_data = list(zip(*buffer))
                                info_feet = {
                                    "right": self.detect_phase(
                                        footswitch_data[emg_num["Right Heel"] - 1],
                                        footswitch_data[emg_num["Right Toe"] - 1],
                                        1,
                                    ),
                                    "left": self.detect_phase(
                                        footswitch_data[emg_num["Left Heel"] - 1],
                                        footswitch_data[emg_num["Left Toe"] - 1],
                                        2,
                                    ),
                                }

                                # Gestion de la stimulation en fonction des états détectés
                                self.manage_stimulation(info_feet)

                    '''if received_data["mks"] and self.visualization_widget.doprocessIK:
                        # TODO add cycle cut and process IK
                        print("todo")'''
                except Exception as e:
                    logging.error(f"Erreur lors de la réception des données: {e}")
                    time.sleep(0.005)  # Optionally wait before retrying


    def detect_phase(self, data_heel, data_toe, foot_num):
        info = "nothing"
        data_heel=data_heel-np.mean(data_heel)
        data_toe = data_toe - np.mean(data_toe)
        if np.mean(np.array(data_heel[:]) ** 2) > 300 and np.mean(np.array(data_toe[:]) ** 2) < 200 and not self.sendStim[foot_num]:
            info = "StartStim"
            self.sendStim[foot_num] = True
        data_toe = data_toe - np.mean(data_toe)
        if np.mean(np.array(data_toe[:]) ** 2) > 300  and self.sendStim[foot_num] is True:
            info = "StopStim"
            self.sendStim[foot_num] = False
        return info

    def manage_stimulation(self, info_feet):
        right = info_feet["right"]
        left = info_feet["left"]

        # Utiliser un set pour éviter les doublons et simplifier les ajouts/suppressions
        active_channels = set(self.last_channels)

        if right == "StartStim":
            active_channels.update([1, 2, 3, 4])
        if left == "StartStim":
            active_channels.update([5, 6, 7, 8])
        if right == "StopStim":
            active_channels.difference_update([1, 2, 3, 4])
        if left == "StopStim":
            active_channels.difference_update([5, 6, 7, 8])

        new_channels = sorted(active_channels)

        # Si changement de canaux, on agit
        if new_channels != self.last_channels:
            if new_channels:
                self.visualization_widget.call_start_stimulation(new_channels)
                print(f"Stim send to canal(s): {new_channels}")
            else:
                self.visualization_widget.call_pause_stimulation()
                print("Stim stop")
            self.last_channels = new_channels
