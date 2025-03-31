import time
from biosiglive import TcpClient
from PyQt5.QtCore import QObject, QTimer
import logging
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from datetime import datetime


class DataReceiver(QObject):
    def __init__(self, server_ip, server_port, visualization_widget, read_frequency=100):
        super().__init__()
        self.visualization_widget = visualization_widget
        self.server_ip = server_ip
        self.server_port = server_port
        self.read_frequency = read_frequency
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=self.read_frequency)
        logging.basicConfig(level=logging.INFO)
        self.sendStim = {1: False, 2: False}
        self.last_channels = []
        self.propulsion_time = None
        self.last_foot_stim = None

        self.event_log = deque(maxlen=500)
        self.fyr_buffer = deque(maxlen=500)
        self.fyl_buffer = deque(maxlen=500)
        self.fzr_buffer = deque(maxlen=500)
        self.fzl_buffer = deque(maxlen=500)
        self.time_buffer = deque(maxlen=500)

        # Plots forces antéropostérieures
        self.force_plot_right = pg.PlotWidget(title="Right AP Force")
        self.plot_curve_fyr = self.force_plot_right.plot(pen='y')

        self.force_plot_left = pg.PlotWidget(title="Left AP Force")
        self.plot_curve_fyl = self.force_plot_left.plot(pen='r')

        # Ajout des plots à la GUI
        self.visualization_widget.layout().addWidget(self.force_plot_right)
        self.visualization_widget.layout().addWidget(self.force_plot_left)

        # Lignes d'événements
        self.event_lines_right = []
        self.event_lines_left = []

        # Timer pour mise à jour régulière
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(100)
        self.period = 0.01

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        # force_data_buffer = [deque(maxlen=30) for _ in range(18)]

        while True:
            start_time = time.perf_counter()
            for _ in range(3):
                try:
                    if self.visualization_widget.dolookneedsendstim:
                        received_data = self.tcp_client.get_data_from_server(command=["force"])

                        if received_data["force"]:
                            force_data_oneframe = received_data["force"]
                            if force_data_oneframe[0].size > 0:
                                timestamp = datetime.now().timestamp()
                                self.time_buffer.append(timestamp)
                                self.fyl_buffer.append(np.mean(force_data_oneframe[1]))
                                self.fzl_buffer.append(np.mean(force_data_oneframe[2]))
                                self.fyr_buffer.append(np.mean(force_data_oneframe[10]))
                                self.fzr_buffer.append(np.mean(force_data_oneframe[11]))

                            if len(self.fyr_buffer) > 60:
                                info_feet = {
                                    "right": self.detect_phase_force(
                                        self.fyr_buffer,
                                        self.fzr_buffer,
                                        1),
                                    "left": self.detect_phase_force(
                                        self.fyl_buffer,
                                        self.fzl_buffer,
                                        2),
                                }
                                self.manage_stimulation(info_feet)

                except Exception as e:
                    logging.error(f"Erreur lors de la réception des données: {e}")

                elapsed = time.perf_counter() - start_time
                to_sleep = self.period - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

    def detect_phase_force(self, data_force_ap, data_force_v, foot_num):
        info = "nothing"
        subject_mass = 500
        fs_camera = 100
        fs_pf = 1000
        ratio = int(fs_pf / fs_camera)
        lastsecond_force_vert =np.array(list(data_force_v)[-60:])

        force_ap_last = data_force_ap[-1]
        force_ap_previous = data_force_ap[-2]
        force_vert_last = data_force_v[-1]

        current_time = datetime.now().timestamp()

        if force_vert_last > 0.7 * subject_mass and  any(lastsecond_force_vert < 50):
            if (force_ap_last < 0.1 * subject_mass
                    and force_ap_previous > force_ap_last
                    and not self.sendStim[foot_num]
                    and self.last_foot_stim is not foot_num):
                info = "StartStim"
                print('heel off')
                self.event_log.append((current_time, f"HeelOff_{foot_num}"))
                self.sendStim[foot_num] = True
                self.last_foot_stim = foot_num

        if (force_vert_last < 0.05 * subject_mass or all(lastsecond_force_vert>50)) and self.sendStim[foot_num]:
            info = "StopStim"
            self.event_log.append((current_time, f"ToeOff_{foot_num}"))
            self.sendStim[foot_num] = False
            print('toe off')

        return info

    def manage_stimulation(self, info_feet):
        right = info_feet["right"]
        left = info_feet["left"]
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

        if new_channels != self.last_channels:
            if new_channels:
                self.visualization_widget.call_start_stimulation(new_channels)
                print(f"Stim send to canal(s): {new_channels}")
            else:
                self.visualization_widget.call_pause_stimulation()
                print("Stim stop")
            self.last_channels = new_channels

    def update_plot(self):
        if not self.event_log or not self.time_buffer:
            return

        times = np.array(self.time_buffer)
        self.plot_curve_fyr.setData(times, self.fyr_buffer)
        self.plot_curve_fyl.setData(times, self.fyl_buffer)

        self.force_plot_right.getPlotItem().getAxis('bottom').setTicks(
            [[(t, datetime.fromtimestamp(t).strftime("%H:%M:%S")) for t in times[::50]]]
        )
        self.force_plot_left.getPlotItem().getAxis('bottom').setTicks(
            [[(t, datetime.fromtimestamp(t).strftime("%H:%M:%S")) for t in times[::50]]]
        )

        self.force_plot_right.getPlotItem().setLabel('left', "Force (N)")
        self.force_plot_right.getPlotItem().setLabel('bottom', "Time")
        self.force_plot_left.getPlotItem().setLabel('left', "Force (N)")
        self.force_plot_left.getPlotItem().setLabel('bottom', "Time")



        # Supprimer anciennes lignes
        for line in self.event_lines_right:
            self.force_plot_right.removeItem(line)
        for line in self.event_lines_left:
            self.force_plot_left.removeItem(line)
        self.event_lines_right.clear()
        self.event_lines_left.clear()

        # Ajouter uniquement les événements visibles
        for timestamp, event in self.event_log:
            if "HeelOff_1" in event or "ToeOff_1" in event:
                if times[0] <= timestamp <= times[-1]:
                    line = pg.InfiniteLine(pos=timestamp, angle=90, movable=False)
                    color = 'g' if "HeelOff" in event else 'b'
                    line.setPen(pg.mkPen(color, width=2))
                    self.force_plot_right.addItem(line)
                    self.event_lines_right.append(line)

            elif "HeelOff_2" in event or "ToeOff_2" in event:
                if times[0] <= timestamp <= times[-1]:
                    line = pg.InfiniteLine(pos=timestamp, angle=90, movable=False)
                    color = 'g' if "HeelOff" in event else 'b'
                    line.setPen(pg.mkPen(color, width=2))
                    self.force_plot_left.addItem(line)
                    self.event_lines_left.append(line)
