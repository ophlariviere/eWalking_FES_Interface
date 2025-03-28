import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from datetime import datetime


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

        # Création des 2 plots (un pour chaque pied)
        self.event_log=deque(maxlen=100)
        self.plot_widget_right = pg.PlotWidget(title="Right Foot Events")
        self.plot_curve_right = self.plot_widget_right.plot([], [], pen=None, symbol='o', symbolBrush='b')

        self.plot_widget_left = pg.PlotWidget(title="Left Foot Events")
        self.plot_curve_left = self.plot_widget_left.plot([], [], pen=None, symbol='o', symbolBrush='r')

        # Ajout des plots à la GUI
        self.visualization_widget.layout().addWidget(self.plot_widget_right)
        self.visualization_widget.layout().addWidget(self.plot_widget_left)

        # Timer pour mise à jour régulière
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(1000)

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        force_data_buffer = [deque(maxlen=30) for _ in range(18)]

        while True:
            for _ in range(3):  # Multiple attempts
                try:
                    if self.visualization_widget.dolookneedsendstim:
                        received_data = self.tcp_client.get_data_from_server(command=["force"])

                        if received_data["force"]:  # Vérifie que des données sont présentes
                            force_data_oneframe = received_data["force"]
                            if force_data_oneframe[0].size > 0:
                                for i in range(18):
                                    force_data_buffer[i].append(force_data_oneframe[i])

                        if all(len(buf) == 30 for buf in force_data_buffer):
                           print('test_phase')
                           info_feet = {
                               "right": self.detect_phase_force(
                               np.concatenate(force_data_buffer[1]),
                               np.concatenate(force_data_buffer[2]),
                               1),
                               "left": self.detect_phase_force(
                                   np.concatenate(force_data_buffer[10]),
                                   np.concatenate(force_data_buffer[11]),
                                   2),
                           }
                           # Gestion de la stimulation (décommenter si tu veux l'activer)
                           self.manage_stimulation(info_feet)

                        '''
                        if received_data["mks"] and self.visualization_widget.doprocessIK:
                            # TODO: découper les cycles et traiter l'IK
                            print("todo")
                        '''

                except Exception as e:
                    logging.error(f"Erreur lors de la réception des données: {e}")
                    time.sleep(0.005)  # Petite pause avant la tentative suivante

                time.sleep(0.01)

    def detect_phase_force(self, data_force_ap, data_force_v, foot_num):
        info = "nothing"
        subject_mass = 500
        fs_camera = 100
        fs_pf = 1000
        ratio = int(fs_pf/fs_camera)
        b, a = butter(2, 10 / (0.5 * fs_pf), btype='low')
        force_ap_filter = filtfilt(b, a, data_force_ap)  # Antéropostérieure
        force_vert_filter = filtfilt(b, a, data_force_v)  # Verticale

        # On prend la dernière valeur pour décider (tu peux aussi faire une moyenne glissante)
        force_ap_last = np.mean(force_ap_filter[-ratio:])  # dernier petit segment
        force_ap_previous = np.mean(force_ap_filter[-2*ratio:-ratio])
        force_vert_last = np.mean(force_vert_filter[-ratio:])

        current_time = datetime.now().timestamp()

        # Si le pied est en appui (grande force verticale)
        if force_vert_last > 0.7 * subject_mass:
            if (force_ap_last < 0.1 * subject_mass
                    and force_ap_previous > force_ap_last
                    and self.sendStim[foot_num] is False):
                info = "StartStim"
                self.event_log.append((current_time, f"HeelOff_{foot_num}"))
                self.sendStim[foot_num] = True
                """
                # Calcul de la dérivée (variation de la force antéropostérieure)
                derive_force_ap = np.diff(force_ap_last)
    
                # Détection de changement de signe (inversion de direction)
                sign_change = np.any(derive_force_ap[:-1] * derive_force_ap[1:] < 0)
    
                if sign_change and not self.sendStim[foot_num]:
                   
                """

        # Si le pied commence à se lever (faible force verticale)
        if force_vert_last < 0.05 * subject_mass and self.sendStim[foot_num]:
            info = "StopStim"
            self.event_log.append((current_time, f"ToeOff_{foot_num}"))
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

    def update_plot(self):
        if not self.event_log:
            return

        times_right, y_right = [0:], []
        times_left, y_left = [], []

        for t, label in self.event_log:
            if "1" in label:  # Foot 1 = Right
                times_right.append(t)
                y_right.append(1 if "HeelOff" in label else 2)
            elif "2" in label:  # Foot 2 = Left
                times_left.append(t)
                y_left.append(1 if "HeelOff" in label else 2)

        self.plot_curve_right.setData(times_right, y_right)
        self.plot_widget_right.getPlotItem().getAxis('bottom').setTicks(
            [[(t, datetime.fromtimestamp(t).strftime("%H:%M:%S")) for t in times_right]]
        )
        self.plot_widget_right.getPlotItem().setYRange(0, 3)
        self.plot_widget_right.getPlotItem().setLabel('left', "Event")
        self.plot_widget_right.getPlotItem().setLabel('bottom', "Time")

        self.plot_curve_left.setData(times_left, y_left)
        self.plot_widget_left.getPlotItem().getAxis('bottom').setTicks(
            [[(t, datetime.fromtimestamp(t).strftime("%H:%M:%S")) for t in times_left]]
        )
        self.plot_widget_left.getPlotItem().setYRange(0, 3)
        self.plot_widget_left.getPlotItem().setLabel('left', "Event")
        self.plot_widget_left.getPlotItem().setLabel('bottom', "Time")
