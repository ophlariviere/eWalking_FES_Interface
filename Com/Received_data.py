import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
import numpy as np
from collections import deque


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
        self.numout = {1: 0, 2: 0}
        self.numin = {1: 0, 2: 0}

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        buffer = deque(maxlen=6)
        while True:
            tic = time.time()
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
                                    "right": self.heel_off_detection(
                                        footswitch_data[emg_num["Right Heel"] - 1],
                                        footswitch_data[emg_num["Right Toe"] - 1],
                                        1,
                                    ),
                                    "left": self.heel_off_detection(
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

    def manage_stimulation(self, info_feet):
        """Gère l'activation ou l'arrêt de la stimulation en fonction des états des pieds."""
        right, left = info_feet["right"], info_feet["left"]

        if right == "StarStim" and left == "StarStim":
            self.visualization_widget.call_start_stimulation([1, 2, 3, 4, 5, 6, 7, 8])
            self.sendStim[1] = True
            self.sendStim[2] = True
            print("Stim send to all canal")
        elif right == "StarStim" and left in ["StopStim", "nothing"]:
            self.visualization_widget.call_start_stimulation([1, 2, 3, 4])
            self.sendStim[1] = True
            self.sendStim[2] = False
            print("Stim send to right")
        elif left == "StarStim" and right in ["StopStim", "nothing"]:
            self.visualization_widget.call_start_stimulation([5, 6, 7, 8])
            self.sendStim[1] = False
            self.sendStim[2] = True
            print("Stim send to left")
        elif right == "StopStim" and left in ["StopStim", "nothing"]:
            self.visualization_widget.call_pause_stimulation()
            self.sendStim[1] = False
            self.sendStim[2] = False
            print("Stim stop")
        elif left == "StopStim" and right in ["StopStim", "nothing"]:
            self.visualization_widget.call_pause_stimulation()
            self.sendStim[1] = False
            self.sendStim[2] = False
            print("Stim stop")

    def heel_off_detection(self, data_foot_switch_heel, data_foot_switch_toe, foot_num):
        #print(f"{round(data_foot_switch_heel)} and {round(data_foot_switch_toe)}")
        info_etat = "nothing"
        print(f"{np.std(data_foot_switch_heel)} and {np.std(data_foot_switch_toe)}")
        if np.std(data_foot_switch_heel) > 500 and np.std(data_foot_switch_toe) < 500 and self.sendStim[foot_num] is False:
                info_etat = "StarStim"

        if np.std(data_foot_switch_heel) > 500 and np.std(data_foot_switch_toe) > 500 and self.sendStim[foot_num] is True:
                info_etat = "StopStim"

        return info_etat
