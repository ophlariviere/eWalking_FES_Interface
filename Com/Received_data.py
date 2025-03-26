import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
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
        self.numout = {1: 0, 2: 0}
        self.numin = {1: 0, 2: 0}

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        while True:
            tic = time.time()
            for _ in range(3):  # Multiple attempts
                try:
                    # Attempt to receive data from the server
                    received_data = self.tcp_client.get_data_from_server(
                        command=["footswitch_data", "force", "mks", "mks_name"]
                    )

                    # Stim gestion
                    if received_data["footswitch_data"]:  # Ensure we have valid data before proceeding
                        footswitch_data = received_data["footswitch_data"]
                        if self.visualization_widget.dolookneedsendstim:
                            emg_num = self.visualization_widget.foot_emg
                            if emg_num:
                                info_feet = {
                                    "right": self.heel_off_detection(
                                        footswitch_data[emg_num["Right Heel"] - 1] ** 2,
                                        footswitch_data[emg_num["Right Toe"] - 1] ** 2,
                                        1,
                                    ),
                                    "left": self.heel_off_detection(
                                        footswitch_data[emg_num["Left Heel"] - 1] ** 2,
                                        footswitch_data[emg_num["Left Toe"] - 1] ** 2,
                                        2,
                                    ),
                                }

                                # Gestion de la stimulation en fonction des états détectés
                                self.manage_stimulation(info_feet)

                    if received_data["mks"] and self.visualization_widget.doprocessIK:
                        # TODO add cycle cut and process IK add interface 
                        print("todo")
                except Exception as e:
                    logging.error(f"Erreur lors de la réception des données: {e}")
                    time.sleep(1)  # Optionally wait before retrying

            loop_time = time.time() - tic
            real_time_to_sleep = max(0, 1 / self.read_frequency - loop_time)
            time.sleep(real_time_to_sleep)

    def manage_stimulation(self, info_feet):
        """Gère l'activation ou l'arrêt de la stimulation en fonction des états des pieds."""
        right, left = info_feet["right"], info_feet["left"]

        if right == "StarStim" and left == "StarStim":
            self.visualization_widget.call_start_stimulation([1, 2, 3, 4, 5, 6, 7, 8])
        elif right == "StarStim" and left in ["StopStim", "nothing"]:
            self.visualization_widget.call_start_stimulation([1, 2, 3, 4])
        elif left == "StarStim" and right in ["StopStim", "nothing"]:
            self.visualization_widget.call_start_stimulation([5, 6, 7, 8])
        elif right == "StopStim" and left in ["StopStim", "nothing"]:
            self.visualization_widget.call_pause_stimulation()
        elif left == "StopStim" and right in ["StopStim", "nothing"]:
            self.visualization_widget.call_pause_stimulation()

    def heel_off_detection(self, data_foot_switch_heel, data_foot_switch_toe, foot_num):
        print(f"{round(data_foot_switch_heel)} and {round(data_foot_switch_toe)}")
        info_etat = "nothing"
        if (data_foot_switch_heel > 360000) and (data_foot_switch_toe < 300) and (self.sendStim[foot_num] is False):
            channel_to_stim = [1, 2, 3, 4] if foot_num == 1 else [5, 6, 7, 8]
            self.numin[foot_num] = self.numin[foot_num] + 1
            if self.numin[foot_num] > 2:
                # self.visualization_widget.call_start_stimulation(channel_to_stim)
                print("Send stim")
                self.numout[foot_num] = 0
                self.sendStim[foot_num] = True
                info_etat = "StarStim"

        if (
            np.nanmean(np.abs(data_foot_switch_heel)) > 300
            and np.nanmean(np.abs(data_foot_switch_toe)) > 360000
            and self.sendStim[foot_num] is True
        ):
            self.numout[foot_num] = self.numout[foot_num] + 1
            if self.numout[foot_num] > 2:
                # self.visualization_widget.call_pause_stimulation()
                print("Stop stim")
                self.numin[foot_num] = 0
                self.sendStim[foot_num] = False
                info_etat = "StopStim"

        return info_etat
