"""
Before starting the code, run in terminal :  docker run --name redis-server -p 6379:6379 -d redis
"""


import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QWidget, QGroupBox, QLabel, QLineEdit, QSpinBox, QComboBox, QFileDialog,
    QMessageBox, QStatusBar
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import redis
import time
import numpy as np
import json
import biorbd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import butter, filtfilt
from pyScienceMode import RehastimP24 as St
from pyScienceMode import Channel, Modes, Device
from biosiglive import TcpClient
import random
from collections import deque
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import threading


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Constantes globales
BUFFER_LENGTH = 800
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

MARKER_FREQUENCY = 100
FORCE_MIN_THRESHOLD = 0.05

# Instance Redis globale
redis_client = None


class RedisConnectionManager:

    def __init__(self):
        super().__init__()
        self.running = True
        global redis_client
        self.connection_status = "Not initialized"

    def run(self):
        global redis_client
        while self.running:
            try:
                redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
                redis_client.flushdb()
                if redis_client.ping():
                    self.connection_status = "Connecté à Redis"
                    logging.info("Connexion Redis réussie")
                    break
                else:
                    self.connection_status = "Échec de connexion: Redis ne répond pas"
            except redis.ConnectionError as e:
                self.connection_status = f"Échec de connexion: {str(e)}"
                logging.warning(f"Échec de connexion Redis: {str(e)}")
            time.sleep(1)

        # Vérification périodique de la connexion
        while self.running:
            try:
                if redis_client and not redis_client.ping():
                    self.connection_status = "Connexion Redis perdue"
                    redis_client = None
            except redis.ConnectionError:
                self.connection_status = "Connexion Redis perdue"
                redis_client = None
            time.sleep(5)  # Vérification toutes les 5 secondes

    def stop(self):
        self.running = False
        self.wait()


def is_redis_connected():
    global redis_client
    try:
        return redis_client and redis_client.ping()
    except:
        return False


def safe_redis_operation(operation, *args, **kwargs):
    if not is_redis_connected():
        logging.warning("Redis n'est pas connecté")
        return None
    try:
        return operation(*args, **kwargs)
    except redis.RedisError as e:
        logging.error(f"Erreur Redis: {str(e)}")
        return None


class DataReceiver:
    """Reçoit les données du serveur TCP et les stocke dans Redis"""

    def __init__(self, server_ip, server_port, read_frequency=100):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.read_frequency = read_frequency
        self.running = False
        self.tcp_client = None
        self.mks_name = None
        self.frame_counter = 0
        self.data_received = "Not initialized"

    def start_receiving(self):
        self.running = True
        try:
            self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=self.read_frequency)

            while self.running:
                try:
                    received_data = self.tcp_client.get_data_from_server(command=["force", "mks", "mks_name"])

                    """ Data markers """
                    if self.mks_name is None:
                        self.mks_name = received_data['mks_name']
                        safe_redis_operation(redis_client.rpush, "mks_name", json.dumps(self.mks_name))
                        safe_redis_operation(redis_client.ltrim, "mks_name", -BUFFER_LENGTH, -1)

                    mks = received_data['mks']
                    nb_markers = len(self.mks_name)
                    markers_frame = np.full((3, nb_markers), np.nan)
                    for i, name in enumerate(self.mks_name):
                        markers_frame[:, i] = mks[i]

                    """ Data forces"""
                    forces_frame = np.full((len(received_data['force']), 9), np.nan)
                    for i in range(len(received_data['force'])):
                        for i2 in range(len(received_data['force'][i])):
                            mean_val = float(np.mean(received_data['force'][i][i2, :]))
                            forces_frame[i][i2] = mean_val

                    # Créer un identifiant unique (timestamp + compteur)
                    # frame_id = f"{time.time()}-{random.randint(1000, 9999)}"
                    frame_id = f"{time.time()}-{self.frame_counter}"
                    self.frame_counter += 1

                    # Stocker l'ID dans une liste séparée pour suivre l'ordre
                    safe_redis_operation(redis_client.rpush, "frame_ids", frame_id)
                    safe_redis_operation(redis_client.ltrim, "frame_ids", -BUFFER_LENGTH, -1)

                    safe_redis_operation(redis_client.rpush, "force", json.dumps(forces_frame.tolist()))
                    safe_redis_operation(redis_client.ltrim, "force",  -BUFFER_LENGTH, -1)
                    safe_redis_operation(redis_client.rpush, "mks", json.dumps(markers_frame.tolist()))
                    safe_redis_operation(redis_client.ltrim, "mks",  -BUFFER_LENGTH, -1)
                    self.data_received = "Data received successfully"

                    # time.sleep(1/self.read_frequency)

                except Exception as e:
                    logging.error(f"Erreur dans DataReceiver: {e}")
                    time.sleep(1)

        except Exception as e:
            logging.error(f"Erreur d'initialisation du client TCP: {e}")

    def stop(self):
        self.running = False
        if self.tcp_client:
            self.tcp_client.close()
        self.wait()


class DataProcessor:
    """Traite les données pour calculer les angles et moments articulaires"""

    def __init__(self):
        super().__init__()
        self.running = True
        self.dof_corr = {
            "LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
            "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
            "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
            "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
            "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)
        }
        self.model = None
        self.processed_frame_ids = deque(maxlen=2*BUFFER_LENGTH)
        self.processing_complete = "Not initialized"
        self.cycle_counter = 0

    def start_processing(self):
        self.running = True

        while self.running:
            try:
                if is_redis_connected():
                    self.process()
                    self.processing_complete = "Processing complete"
                # time.sleep(0.1)  # Réduire la fréquence de traitement
            except Exception as e:
                logging.error(f"Erreur dans DataProcessor: {e}")
                time.sleep(1)

    def identify_cycle_start(self, forces_all):
        print("Identifying cycle start...")
        force_filtered = self.data_filter(forces_all[0, 0:3, :], 2, MARKER_FREQUENCY, 10)
        subject_mass = float(safe_redis_operation(redis_client.lrange, "participant_mass", -1, -1)[0])
        current_cycle_idx = np.ones((forces_all[0].shape[1], )) * self.cycle_counter

        right_foot_on_ground_idx = force_filtered[2, :] > FORCE_MIN_THRESHOLD*2 * subject_mass
        right_foot_on_ground_idx = np.astype(right_foot_on_ground_idx, int)
        heel_strike_idx = np.where(np.diff(right_foot_on_ground_idx) == 1)[0] + 1
        toe_off_idx = np.where(np.diff(right_foot_on_ground_idx) == -1)[0] + 1

        # Identification : OK
        plt.figure()
        plt.plot(force_filtered[2, :])
        for frame in heel_strike_idx:
            plt.axvline(x=frame, color='red', linestyle='--', label='Heel Strike')
        for frame in toe_off_idx:
            plt.axvline(x=frame, color='green', linestyle='--', label='Toe Off')
        plt.savefig("cycle_identification.png")
        plt.show()

        if heel_strike_idx > 1 or toe_off_idx > 1:
            raise RuntimeError("There was more than one heel strike in this cycle")
        if right_foot_on_ground_idx[0] == False and len(heel_strike_idx) > 0:
            self.cycle_counter += 1
            current_cycle_idx[heel_strike_idx:toe_off_idx] = self.cycle_counter
        print(current_cycle_idx)

    def process(self):
        try:
            # Récupérer les IDs des frames disponibles
            frame_ids = [x.decode('utf-8') for x in redis_client.lrange("frame_ids", 0, -2)]

            # Filtrer pour ne garder que les nouveaux IDs
            new_indices = [i for i, frame_id in enumerate(frame_ids) if frame_id not in self.processed_frame_ids]
            new_frame_ids = [frame_id for frame_id in frame_ids if frame_id not in self.processed_frame_ids]
            new_frame_ids = np.array(new_frame_ids)
            new_indices = np.array(new_indices)
            if not new_indices.any():
                return  # Rien de nouveau à traiter
            if len(new_frame_ids) > 400:
                print([new_frame_ids[0], len(new_frame_ids), new_frame_ids[-1]])
                forces_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)
                mks_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("mks", 0, -1)]
                mks_all = np.array(mks_all).transpose(1, 2, 0)

                self.identify_cycle_start(forces_all)

                # Récupérer les données pour ces nouveaux IDs
                print(mks_all.shape)
                print(forces_all.shape)
                print(new_indices)
                mks = np.take(mks_all, new_indices, axis=2)
                forces = np.take(forces_all, new_indices, axis=2)

                mks_name = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("mks_name", 0, -1)]
                # self.identify_cycle_start(forces_all)

                self.processed_frame_ids.extend(new_frame_ids)

                file_name = redis_client.lrange("model_file_name", 0, -1)
                if file_name:
                    file_name2 = file_name[0].decode('utf-8')
                    self.model = biorbd.Model(file_name2)

                if self.model is not None:
                    # Calculer IK/ID
                    print("Calcul IK/ID...")
                    q, qdot, qddot = self.calculate_ik(self.model, mks, mks_name[0])
                    tau = self.calculate_id(self.model, forces, q, qdot, qddot)

                    # Stocker les résultats dans Redis
                    if q is not None and tau is not None:
                        for i in range(tau.shape[1]):
                            safe_redis_operation(redis_client.rpush, "q", json.dumps(q[:, i].tolist()))
                            safe_redis_operation(redis_client.rpush, "tau", json.dumps(tau[:, i].tolist()))

                        safe_redis_operation(redis_client.ltrim, "q", -BUFFER_LENGTH, -1)
                        safe_redis_operation(redis_client.ltrim, "tau", -BUFFER_LENGTH, -1)
        except Exception as e:
            logging.error(f"Erreur lors du traitement des données: {e}")

    def calculate_ik(self, model, mks, labels):
        try:
            n_frames = mks.shape[2]
            marker_names = tuple(n.to_string() for n in self.model.technicalMarkerNames())
            index_in_c3d = np.array(tuple(labels.index(name) if name in labels else -1 for name in marker_names))
            markers_in_c3d = np.ndarray((3, len(index_in_c3d), n_frames)) * np.nan
            mks_to_filter = mks[:3, index_in_c3d[index_in_c3d >= 0], :]
            # Apply the filter to each coordinate (x, y, z) over time
            smoothed_mks = self.data_filter(data=mks_to_filter, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)

            # Store the result
            markers_in_c3d[:, index_in_c3d >= 0, :] = smoothed_mks
            ik = biorbd.InverseKinematics(model, markers_in_c3d)
            ik.solve(method="trf")
            q = ik.q
            q = self.data_filter(q, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)
            qdot = np.gradient(q, axis=1) * MARKER_FREQUENCY
            qddot = np.gradient(qdot, axis=1) * MARKER_FREQUENCY
            return q, qdot, qddot
        except Exception as e:
            logging.error(f"Erreur dans calculate_ik: {e}")
            return None, None, None

    def calculate_id(self, model, force, q, qdot, qddot):
        try:
            num_contacts = len(force)
            num_frames = force[0].shape[1]
            platform_origin = [[0.78485, 0.7825, 0.], [0.78485, 0.2385, 0.]]
            force_filtered = np.zeros((num_contacts, 3, num_frames))
            moment_filtered = np.zeros((num_contacts, 3, num_frames))
            tau_data = np.zeros((model.nbQ(), num_frames))

            for contact_idx in range(num_contacts):
                force_filtered[contact_idx] = self.data_filter(force[contact_idx][0:3], 2, MARKER_FREQUENCY, 10)
                moment_filtered[contact_idx] = self.data_filter(force[contact_idx][3:6], 4, MARKER_FREQUENCY, 10)

            for i in range(num_frames):
                ext_load = model.externalForceSet()
                for contact_idx in range(num_contacts):
                    fz = force_filtered[contact_idx, 2, i]
                    if fz > 30:
                        force_vec = force_filtered[contact_idx, :, i]
                        moment_vec = moment_filtered[contact_idx, :, i]/1000
                        spatial_vector = np.concatenate((moment_vec, force_vec))
                        point_app = platform_origin[contact_idx]
                        segment_name = "LFoot" if contact_idx == 0 else "RFoot"
                        ext_load.add(biorbd.String(segment_name), spatial_vector, np.array(point_app))

                tau = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], ext_load)
                tau_data[:, i] = tau.to_array()

            return tau_data
        except Exception as e:
            logging.error(f"Erreur dans calculate_id: {e}")
            return None

    def data_filter(self, data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low')

        data = np.asarray(data)
        filtered_data = np.empty_like(data)

        if data.ndim == 2:  # (3, T)
            for i in range(data.shape[0]):
                filtered_data[i, :] = self.nan_filtfilt(b, a, data[i, :])
        elif data.ndim == 3:  # (3, N, T)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    filtered_data[i, j, :] = self.nan_filtfilt(b, a, data[i, j, :])
        else:
            raise ValueError("Data must be 2D or 3D.")

        return filtered_data

    @staticmethod
    def nan_filtfilt(b, a, data):
        nan_mask = np.isnan(data)
        if np.all(nan_mask):
            return np.zeros_like(data)

        filtered = np.copy(data)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 1:
            filtered[valid_idx] = filtfilt(b, a, data[valid_idx])
        return filtered

    def stop(self):
        self.running = False
        self.wait()


class StimulationProcessor:

    def __init__(self):
        super().__init__()
        self.running = True
        self.stimulator = None
        self.stimulator_is_active = False
        self.stimulator_is_sending_stim = False
        self.sendStim = {1: False, 2: False}
        self.last_foot_stim = None
        self.last_channels = []
        self.processed_frame_ids = deque(maxlen=2*BUFFER_LENGTH)
        self.data_received = "Not initialized"

    def start_processing(self):
        while self.running:
            try:
                if is_redis_connected():
                    self.stimulation_process()
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Erreur dans StimulationProcessor: {e}")
                time.sleep(0.01)

    def stimulation_process(self):
        try:
            frame_ids = [x.decode('utf-8') for x in redis_client.lrange("frame_ids", 0, -1)]
            new_indices = [i for i, frame_id in enumerate(frame_ids) if frame_id not in self.processed_frame_ids]
            new_frame_ids = [frame_id for frame_id in frame_ids if frame_id not in self.processed_frame_ids]
            new_frame_ids = np.array(new_frame_ids)
            new_indices = np.array(new_indices)
            new_indices = new_indices[new_indices < BUFFER_LENGTH-1]
            if not new_indices.any():
                return  # Rien de nouveau à traiter

            forces_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("force", 0, -1)]
            forces_all = np.array(forces_all).transpose(1, 2, 0)
            force_data = np.take(forces_all, new_indices, axis=2)
            self.processed_frame_ids.extend(new_frame_ids)

            # Récupérer les paramètres de stimulation
            stim_params = safe_redis_operation(redis_client.lrange, "stimulation_parameters", 0, -1)
            if stim_params:
                stimulator_parameters = json.loads(stim_params[-1])
                subject_mass = float(safe_redis_operation(redis_client.lrange, "participant_mass", -1, -1)[0])
                if len(force_data) > 0:
                    fyr = force_data[0][2]  # Force Y droite
                    fzr = force_data[0][5]  # Force Z droite
                    fyl = force_data[0][8]  # Force Y gauche
                    fzl = force_data[0][11]  # Force Z gauche
                    if len(fyr) > 20 and subject_mass:
                        info_feet = {
                            "right": self.detect_phase_force(fyr, fzr, fzl, 1, subject_mass),
                            "left": self.detect_phase_force(fyl, fzl, fzr, 2, subject_mass),
                        }
                        self.manage_stimulation(info_feet, stimulator_parameters)
        except Exception as e:
            logging.error(f"Erreur dans stimulation_process: {e}")

    def detect_phase_force(self, data_force_ap, data_force_v, data_force_opp, foot_num, subject_mass):
        try:
            info = "nothing"
            last_second_force_vert = np.array(list(data_force_opp)[-30:])

            force_ap_last = data_force_ap[-1]
            # force_ap_previous = data_force_ap[-2]
            force_vert_last = data_force_v[-1]
            deri = data_force_ap[1:]-data_force_ap[0:-1]  # last-previous

            if force_vert_last > 0.7 * subject_mass and np.any(last_second_force_vert > 50):
                if (force_ap_last < 0.1 * subject_mass
                        and np.any(deri < 0)  # force_ap_previous > force_ap_last
                        and not self.sendStim[foot_num]
                        and self.last_foot_stim is not foot_num):
                    info = "StartStim"
                    self.sendStim[foot_num] = True
                    self.last_foot_stim = foot_num

            if ((force_vert_last < FORCE_MIN_THRESHOLD * subject_mass
                 or (np.any(deri.any > 0)  # force_ap_previous < force_ap_last
                     and force_ap_last > -0.01 * subject_mass))
                    and self.sendStim[foot_num]):
                info = "StopStim"
                self.sendStim[foot_num] = False

            return info
        except Exception as e:
            logging.error(f"Erreur dans detect_phase_force: {e}")
            return "nothing"

    def manage_stimulation(self, info_feet, stimulator_parameters):
        try:
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
                    self.call_start_stimulation(new_channels, stimulator_parameters)
                    self.stimulation_status = f"Stim send to canal(s): {new_channels}"
                else:
                    self.call_pause_stimulation()
                    self.stimulation_status = "Stim stop"
                self.last_channels = new_channels
        except Exception as e:
            logging.error(f"Erreur dans manage_stimulation: {e}")

    def activate_stimulator(self):
        try:
            if not self.stimulator_is_active:
                self.stimulator = St(port="COM3", show_log="Status")
                self.stimulator_is_active = True
                self.stimulation_status = "Stimulateur activé"
        except Exception as e:
            logging.error(f"Erreur lors de l'activation du stimulateur: {e}")
            self.stimulation_status = f"Erreur: {str(e)}"

    def call_start_stimulation(self, channel_to_send, stimulator_parameters):
        try:
            if not self.stimulator_is_active:
                self.activate_stimulator()
                time.sleep(1)  # Attendre que le stimulateur soit prêt

            if self.stimulator_is_sending_stim:
                self.call_pause_stimulation()

            channels_instructions = [
                Channel(
                    no_channel=channel,
                    name=stimulator_parameters[str(channel)]["name"],
                    amplitude=stimulator_parameters[str(channel)]["amplitude"] if channel in channel_to_send else 0,
                    pulse_width=stimulator_parameters[str(channel)]["pulse_width"],
                    frequency=stimulator_parameters[str(channel)]["frequency"],
                    mode=getattr(Modes, stimulator_parameters[str(channel)]["mode"]),
                    device_type=Device.Rehastimp24,
                )
                for channel in stimulator_parameters
            ]

            if channels_instructions:
                self.stimulator.init_stimulation(list_channels=channels_instructions)
                self.stimulator.update_stimulation(upd_list_channels=channels_instructions)
                self.stimulator.start_stimulation(upd_list_channels=channels_instructions)
                self.stimulator_is_sending_stim = True
                self.stimulation_status = f"Stimulation démarrée sur les canaux {channel_to_send}"
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de la stimulation: {e}")
            self.stimulation_status = f"Erreur stimulation: {str(e)}"

    def call_pause_stimulation(self):
        try:
            if self.stimulator and self.stimulator_is_sending_stim:
                self.stimulator.end_stimulation()
                self.stimulator_is_sending_stim = False
                self.stimulation_status = "Stimulation arrêtée"
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt de la stimulation: {e}")

    def stop_stimulator(self):
        try:
            if self.stimulator:
                self.call_pause_stimulation()
                self.stimulator.close_port()
                self.stimulator_is_active = False
                self.stimulator = None
                self.stimulation_status = "Stimulateur arrêté"
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt du stimulateur: {e}")

    def stop(self):
        self.running = False
        self.stop_stimulator()
        self.wait()


class Interface(QMainWindow):
    """Interface principale de l'application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Système de Stimulation Neuromusculaire")
        self.setMinimumSize(800, 600)
        self.channel_inputs = {}
        self.num_config = 0
        self.mass = 70
        self.process_idik = False
        self.dolookneedsendstim = False
        self.model = None
        self.DataToPlot = self.initialize_data_to_plot()

        # Redis manager
        self.redis_manager = RedisConnectionManager()

        # Data receiver (goal: interaction with Qualisys)
        self.data_receiver = DataReceiver("127.0.0.1", 50000)

        # Data processor (goal: ID, IK)
        self.data_processor = DataProcessor()

        # Stimulation processor (goal: determine if a stim is needed + interaction with stimulator)
        self.stimulation_processor = StimulationProcessor()

        # Initialize UI components
        self.init_ui()

        # # Timer pour mettre à jour les graphes toutes les secondes
        # self.graph_update_timer = QTimer(self)
        # self.graph_update_timer.timeout.connect(self.update_data_and_graphs)
        # self.graph_update_timer.start(100)

        # --- Thread activation --- #
        threading.Thread(target=self.redis_manager.run, daemon=False).start()
        threading.Thread(target=self.data_receiver.start_receiving, daemon=False).start()
        threading.Thread(target=self.data_processor.start_processing, daemon=False).start()
        threading.Thread(target=self.stimulation_processor.start_processing, daemon=False).start()


    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        event.accept()

    def init_ui(self):
        """Initialise l'interface utilisateur"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Barre de statut
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.connection_status = QLabel("Statut: Non connecté")
        self.connection_status.setStyleSheet("color: red;")
        self.stimulation_status = QLabel("Stimulation: Inactive")
        self.stimulation_status.setStyleSheet("color: gray;")
        self.status_bar.addWidget(self.connection_status)
        self.status_bar.addWidget(self.stimulation_status)

        # Configuration des informations du participant
        main_layout.addWidget(self.create_participant_info())

        # Configuration des canaux de stimulation
        main_layout.addWidget(self.create_channel_config_group())

        # Contrôles de stimulation
        main_layout.addLayout(self.create_stimulation_controls())

        main_layout.addWidget(self.create_analysis_group())

        # Zone graphique
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def create_participant_info(self):
        """Crée le groupe d'informations du participant"""
        groupbox = QGroupBox("Participant Info")
        main_layout = QVBoxLayout()

        # Masse
        mass_layout = QHBoxLayout()
        mass_label = QLabel("Masse [kg]:")
        self.mass_spin = QSpinBox()
        self.mass_spin.setRange(0, 400)
        self.mass_spin.setValue(70)
        ok_mass = QPushButton("OK")
        ok_mass.clicked.connect(lambda: self.update_mass(self.mass_spin.value()))
        mass_layout.addWidget(mass_label)
        mass_layout.addWidget(self.mass_spin)
        mass_layout.addWidget(ok_mass)

        # Modèle
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Aucun fichier sélectionné")
        model_button = QPushButton("Charger un fichier")
        model_button.clicked.connect(self.upload_file)
        self.checkbox_pro_idik = QCheckBox("Processus IK/ID")
        self.checkbox_pro_idik.setChecked(False)
        self.checkbox_pro_idik.setEnabled(False)
        self.checkbox_pro_idik.stateChanged.connect(self.need_process_idik)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(model_button)
        model_layout.addWidget(self.checkbox_pro_idik)

        main_layout.addLayout(mass_layout)
        main_layout.addLayout(model_layout)
        groupbox.setLayout(main_layout)
        return groupbox

    def need_process_idik(self):
        """Active le traitement IK/ID"""
        self.process_idik = self.checkbox_pro_idik.isChecked()

    def upload_file(self):
        """Charge un fichier de modèle"""
        if not is_redis_connected():
            QMessageBox.warning(self, "Erreur", "La connexion Redis n'est pas établie. Veuillez patienter...")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier")
        if file_name:
            try:
                safe_redis_operation(redis_client.rpush, "model_file_name", file_name)
                safe_redis_operation(redis_client.ltrim, "model_file_name",  -BUFFER_LENGTH, -1)
                logging.info(f"Fichier modèle chargé: {file_name}")
                self.model_label.setText(file_name.split('/')[-1])
            except Exception as e:
                logging.error(f"Erreur lors du chargement du fichier: {str(e)}")
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier: {str(e)}")

    def update_mass(self, mass_value):
        """Met à jour la masse du participant"""
        self.mass = float(mass_value)
        if not is_redis_connected():
            QMessageBox.warning(self, "Erreur", "La connexion Redis n'est pas établie. Veuillez patienter...")
            return

        try:
            safe_redis_operation(redis_client.rpush, "participant_mass", str(self.mass))
            safe_redis_operation(redis_client.ltrim, "participant_mass",  -BUFFER_LENGTH, -1)
            logging.info(f"Masse mise à jour: {self.mass} kg")
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour de la masse: {str(e)}")
            QMessageBox.critical(self, "Erreur", f"Impossible de mettre à jour la masse: {str(e)}")

    def create_channel_config_group(self):
        """Crée le groupe de configuration des canaux"""
        groupbox = QGroupBox("Configurer les canaux")
        layout = QVBoxLayout()

        # Checkbox pour appliquer à tous les canaux
        self.copy_to_all_checkbox = QCheckBox("Appliquer à tous les canaux")
        self.copy_to_all_checkbox.setToolTip("Utilise les réglages du premier canal sélectionné pour tous les autres")
        self.copy_to_all_checkbox.stateChanged.connect(self.apply_same_settings_to_all_channels)
        layout.addWidget(self.copy_to_all_checkbox)

        # Ajouter les cases à cocher pour sélectionner les canaux
        self.checkboxes = []
        checkbox_layout = QHBoxLayout()
        for i in range(1, 9):
            checkbox = QCheckBox(f"Canal {i}")
            checkbox.stateChanged.connect(self.update_channel_inputs)
            checkbox_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        layout.addLayout(checkbox_layout)

        # Layout pour les configurations de canaux
        self.channel_config_layout = QVBoxLayout()
        layout.addLayout(self.channel_config_layout)

        groupbox.setLayout(layout)
        return groupbox

    def update_channel_inputs(self):
        """Met à jour les entrées des canaux sélectionnés"""
        selected_channels = [i + 1 for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

        # Supprimer les canaux désélectionnés
        for channel in list(self.channel_inputs.keys()):
            if channel not in selected_channels:
                inputs = self.channel_inputs.pop(channel)
                layout = inputs["layout"]
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                self.channel_config_layout.removeItem(layout)

        # Ajouter les nouveaux canaux sélectionnés
        for channel in selected_channels:
            if channel not in self.channel_inputs:
                channel_layout = QHBoxLayout()

                # Création des widgets d'entrée pour le canal
                name_input = QLineEdit()
                name_input.setPlaceholderText(f"Nom du canal {channel}")
                amplitude_input = QSpinBox()
                amplitude_input.setRange(0, 100)
                amplitude_input.setSuffix(" mA")
                pulse_width_input = QSpinBox()
                pulse_width_input.setRange(0, 1000)
                pulse_width_input.setSuffix(" µs")
                frequency_input = QSpinBox()
                frequency_input.setRange(0, 200)
                frequency_input.setSuffix(" Hz")
                mode_input = QComboBox()
                mode_input.addItems(["SINGLE", "DOUBLET", "TRIPLET"])

                channel_layout.addWidget(QLabel(f"Canal {channel}:"))
                channel_layout.addWidget(name_input)
                channel_layout.addWidget(amplitude_input)
                channel_layout.addWidget(pulse_width_input)
                channel_layout.addWidget(frequency_input)
                channel_layout.addWidget(mode_input)

                self.channel_config_layout.addLayout(channel_layout)

                self.channel_inputs[channel] = {
                    "layout": channel_layout,
                    "name_input": name_input,
                    "amplitude_input": amplitude_input,
                    "pulse_width_input": pulse_width_input,
                    "frequency_input": frequency_input,
                    "mode_input": mode_input,
                }

    def create_stimulation_controls(self):
        """Crée les contrôles de stimulation"""
        layout = QHBoxLayout()

        self.activate_button = QPushButton("Activer Stimulateur")
        self.activate_button.clicked.connect(self.stimulation_processor.activate_stimulator)

        self.update_button = QPushButton("Actualiser Paramètres")
        self.update_button.clicked.connect(self.update_stimulation_parameter)

        self.start_button = QPushButton("Envoyer Stimulation")
        self.start_button.clicked.connect(self.stimulation_processor.call_start_stimulation)

        self.stop_button = QPushButton("Arrêter Stimulateur")
        self.stop_button.clicked.connect(self.stimulation_processor.stop_stimulator)

        self.checkpauseStim = QCheckBox("Stop tying send stim")
        self.checkpauseStim.setChecked(True)
        self.checkpauseStim.stateChanged.connect(self.pausefonctiontosendstim)

        layout.addWidget(self.checkpauseStim)
        layout.addWidget(self.activate_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.update_button)
        layout.addWidget(self.stop_button)

        return layout

    def pausefonctiontosendstim(self):
        """Met à jour l'état d'envoi de stimulation"""
        self.dolookneedsendstim = not self.checkpauseStim.isChecked()

    def apply_same_settings_to_all_channels(self):
        """Applique les mêmes paramètres à tous les canaux"""
        if not self.copy_to_all_checkbox.isChecked():
            return

        selected_channels = list(self.channel_inputs.keys())
        if len(selected_channels) < 2:
            return

        # Utiliser le premier canal sélectionné comme référence
        ref = self.channel_inputs[selected_channels[0]]
        for ch in selected_channels[1:]:
            self.channel_inputs[ch]["name_input"].setText(ref["name_input"].text())
            self.channel_inputs[ch]["amplitude_input"].setValue(ref["amplitude_input"].value())
            self.channel_inputs[ch]["pulse_width_input"].setValue(ref["pulse_width_input"].value())
            self.channel_inputs[ch]["frequency_input"].setValue(ref["frequency_input"].value())
            self.channel_inputs[ch]["mode_input"].setCurrentIndex(ref["mode_input"].currentIndex())

    def update_stimulation_parameter(self):
        """Met à jour les paramètres de stimulation"""
        self.num_config += 1
        stimulator_parameters = {}

        for channel, inputs in self.channel_inputs.items():
            stimulator_parameters[str(channel)] = {
                "name": inputs["name_input"].text(),
                "amplitude": inputs["amplitude_input"].value(),
                "pulse_width": inputs["pulse_width_input"].value(),
                "frequency": inputs["frequency_input"].value(),
                "mode": inputs["mode_input"].currentText(),
            }

        if RedisConnectionManager.r and RedisConnectionManager.redis_connected:
            try:
                safe_redis_operation(redis_client.rpush, "stimulation_parameters", json.dumps(stimulator_parameters))
                safe_redis_operation(redis_client.ltrim, "stimulation_parameters",  -BUFFER_LENGTH, -1)
                logging.info("Paramètres de stimulation mis à jour")
            except Exception as e:
                logging.error(f"Erreur lors de la mise à jour des paramètres: {e}")

    def update_connection_status(self, connected, message):
        """Met à jour le statut de connexion"""
        self.connection_status.setText(f"Statut: {message}")
        self.connection_status.setStyleSheet("color: green;" if connected else "color: red;")

    def update_stimulation_status(self, message):
        """Met à jour le statut de stimulation"""
        self.stimulation_status.setText(f"Stimulation: {message}")
        self.stimulation_status.setStyleSheet(
            "color: green;" if "démarrée" in message.lower() or "active" in message.lower()
            else "color: red;" if "arrêt" in message.lower() or "erreur" in message.lower()
            else "color: gray;"
        )

    @staticmethod
    def initialize_data_to_plot():
        """Initialise le dictionnaire des données à tracer."""
        keys = [
            "force_1", "force_2",
            "tau_LHip", "tau_LKnee", "tau_LAnkle",
            "q_LHip", "q_LKnee", "q_LAnkle",
        ]
        return {key: {} for key in keys}

    def create_analysis_group(self):
        """Créer un groupbox pour la sélection des analyses."""
        groupbox = QGroupBox("Sélections d'Analyse")
        layout = QHBoxLayout()

        self.checkboxes_graphs = {}
        for key in self.DataToPlot.keys():
            checkbox = QCheckBox(key, self)
            checkbox.stateChanged.connect(self.update_graphs)
            layout.addWidget(checkbox)
            self.checkboxes_graphs[key] = checkbox

        groupbox.setLayout(layout)
        return groupbox

    def update_data_and_graphs(self):
        # Parcours des clés de self.DataToPlot
        for key in self.DataToPlot.keys():
            if 'force' in key:
                data = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("force", 0, -1)]
                data = np.array(data).transpose(1, 2, 0)
                if data is None:
                    continue
                if 'force_1' in key:
                    self.DataToPlot[key] = data[0][2, :]
                if 'force_2' in key:
                    self.DataToPlot[key] = data[1][2, :]
            else:
                data_l = None
                if 'tau' in key:
                    data_l = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("tau", 0, -1)]

                elif 'q' in key:
                    data_l = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("q", 0, -1)]

                if not data_l:
                    continue

                n_frames = len(data_l)
                n_dof = len(data_l[0])
                data = np.zeros((n_dof, n_frames))
                for frame_i, frame in enumerate(data_l):  # Chaque frame est une liste de 8 valeurs
                    for dof_i, dof in enumerate(frame):
                        data[dof_i, frame_i] = dof
                if data is None:
                    continue

                if 'q' in key:
                    data = data * 180 / np.pi

                if 'LHip' in key:
                    self.DataToPlot[key] = data[37, :]
                elif 'LAnkle' in key:
                    self.DataToPlot[key] = data[40, :]
                elif 'Lknee' in key:
                    self.DataToPlot[key] = data[43, :]
        self.update_graphs()

    def update_graphs(self):
        """Updates displayed graphs based on selected checkboxes."""
        self.figure.clear()

        # Check selected graphs
        graphs_to_display = {key: checkbox.isChecked() for key, checkbox in self.checkboxes_graphs.items()}
        count = sum(graphs_to_display.values())

        if count == 0:
            # Nothing to display
            self.canvas.draw()
            return

        # Calculate layout for subplots
        rows = (count + 1) // 2
        cols = 2 if count > 1 else 1
        subplot_index = 1

        # Affichage des graphiques en fonction des cases à cocher
        for key, is_checked in graphs_to_display.items():
            if is_checked:
                # Ajouter un sous-graphe pour chaque graphique sélectionné
                ax = self.figure.add_subplot(rows, cols, subplot_index)
                if key in self.DataToPlot.keys():
                    self.plot_vector_data(ax, key)
                subplot_index += 1

        # Redessiner le canevas pour afficher les nouvelles données
        self.canvas.draw()

    def plot_vector_data(self, ax, key):
        data = self.DataToPlot[key]
        ax.plot(data)
        ax.set_xlabel('Frame')
        ax.set_ylabel(key)

    @staticmethod
    def on_data_received():
        """Gère la réception de nouvelles données"""
        logging.debug("Nouvelles données reçues")

    @staticmethod
    def on_processing_complete():
        """Gère la fin du traitement des données"""
        logging.debug("Traitement des données terminé")


def main():
    """Point d'entrée principal"""

    # GUI (goal: interaction with the user)
    app = QApplication(sys.argv)
    interface = Interface()
    interface.show()
    sys.exit(app.exec_()) #  Start the GUI


if __name__ == "__main__":
    main()