import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt


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
        buffer = deque(maxlen=100)
        while True:
            for _ in range(3):  # Multiple attempts
                try:
                    # Attempt to receive data from the server
                    received_data = self.tcp_client.get_data_from_server(
                        command=["force"]
                    )

                    # Stim gestion
                    if received_data["force"]:  # Ensure we have valid data before proceeding
                        force_data_oneframe = received_data["force"]
                        buffer.append(force_data_oneframe)
                        if self.visualization_widget.dolookneedsendstim:
                            force_data = list(zip(*buffer))
                            info_feet = {
                                "right": self.detect_phase_force(force_data[1], force_data[2], 1),
                                "left": self.detect_phase_force(force_data[10], force_data[11], 2),
                            }

                            # Gestion de la stimulation en fonction des états détectés
                            self.manage_stimulation(info_feet)

                    '''if received_data["mks"] and self.visualization_widget.doprocessIK:
                        # TODO add cycle cut and process IK
                        print("todo")'''
                except Exception as e:
                    logging.error(f"Erreur lors de la réception des données: {e}")
                    time.sleep(0.005)  # Optionally wait before retrying

    def detect_phase_force(self, data_force_ap, data_force_v, foot_num):
        info = "nothing"
        subject_mass = 500
        fs_camera = 100
        fs_pf = 1000
        ratio = fs_pf/fs_camera
        b, a = butter(2, 10 / (0.5 * fs_pf), btype='low')
        force_ap_filter = filtfilt(b, a, data_force_ap)  # Antéropostérieure
        force_vert_filter = filtfilt(b, a, data_force_v)  # Verticale

        # On prend la dernière valeur pour décider (tu peux aussi faire une moyenne glissante)
        force_ap_last = np.mean(force_ap_filter[-ratio:])  # dernier petit segment
        force_ap_previous = np.mean(force_ap_filter[-2*ratio:-ratio])
        force_vert_last = np.mean(force_vert_filter[-ratio:])

        # Si le pied est en appui (grande force verticale)
        if force_vert_last > 0.7 * subject_mass:
            if (force_ap_last < 0.1 * subject_mass
                    and force_ap_previous > force_ap_last
                    and self.sendStim[foot_num] is False):

                """
                # Calcul de la dérivée (variation de la force antéropostérieure)
                derive_force_ap = np.diff(force_ap_last)
    
                # Détection de changement de signe (inversion de direction)
                sign_change = np.any(derive_force_ap[:-1] * derive_force_ap[1:] < 0)
    
                if sign_change and not self.sendStim[foot_num]:
                    info = "StartStim"
                    self.sendStim[foot_num] = True
                """

        # Si le pied commence à se lever (faible force verticale)
        if force_vert_last < 0.05 * subject_mass and self.sendStim[foot_num]:
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
