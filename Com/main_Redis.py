"""
Before starting the code, run in terminal :  docker run --name redis-server -p 6379:6379 -d redis

Info:
The Redis database contains
1. Data at each frame (force, mks, mks_names, frame_ids)
2. Data at each cycle (q, tau, cycle_ids)
"""


import sys
from enum import Enum
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QWidget, QGroupBox, QLabel, QLineEdit, QSpinBox, QComboBox, QFileDialog,
    QMessageBox, QStatusBar, QGridLayout, QRadioButton,
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
from matplotlib import colormaps as cm
import threading
from skopt import gp_minimize
from skopt.space import Real


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Constantes globales
FRAME_BUFFER_LENGTH = 800
CYCLE_BUFFER_LENGTH = 100
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

MARKER_FREQUENCY = 100
FORCE_MIN_THRESHOLD = 0.05

# Flags to check for stimulation and processing
ACTIVATE_STIMULATOR = False
START_STIMULATION = False
STOP_STIMULATOR = False

# Single value shared variables (instead of duplicating the information in the redis database)
MASS = 70  # Initial value only (will be set by Interface.update_mass)
MODEL_FILE_NAME = None  # Will be set by Interface.upload_file
MODEL = None  # Will be set by Interface.upload_file
NB_DOF = 45
DEFAULT_BOUNDS = {
    "Amplitude": [0, 100],  # Amplitude en mA
    "Pulse Width": [0, 1000],  # Largeur d'impulsion en microsecondes
    "Frequency": [0, 200],  # Fréquence en Hz
}

# Instance Redis globale
redis_client = None


class StimulationMode(Enum):
    MANUAL = "manual"
    BAYESIAN = "bayesian"
    ILC = "ilc"



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
                redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
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
            except redis.ConnectionError as e:
                self.connection_status = "Connexion Redis perdue"
                logging.warning(f"Échec de connexion Redis: {str(e)}")
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


def get_new_indices(processed_frame_ids, print_option=False):
    """ Filtrer pour ne garder que les nouveaux IDs """
    global redis_client

    try:
        frame_ids = [x.decode('utf-8') for x in redis_client.lrange("frame_ids", 0, -1)]
        new_indices = [i for i, frame_id in enumerate(frame_ids) if frame_id not in processed_frame_ids]
        new_frame_ids = [frame_id for frame_id in frame_ids if frame_id not in processed_frame_ids]
        new_frame_ids = np.array(new_frame_ids)
        new_indices = np.array(new_indices)

        if print_option:
            if len(processed_frame_ids) > 0:
                print("processed ", processed_frame_ids[-1])
            print("frame ids ", frame_ids[0], frame_ids[-1])
            print("new indices ", new_indices[0], new_indices[-1])
    except:
        logging.error("erreur lors de lidentification des new indices.")

    return new_indices, new_frame_ids, frame_ids


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
        self.frame_counter = -1
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
                        safe_redis_operation(redis_client.ltrim, "mks_name", -FRAME_BUFFER_LENGTH, -1)

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
                    self.frame_counter += 1
                    frame_id = f"{time.time()}-{self.frame_counter}"

                    # Stocker l'ID dans une liste séparée pour suivre l'ordre
                    safe_redis_operation(redis_client.rpush, "frame_ids", frame_id)
                    safe_redis_operation(redis_client.ltrim, "frame_ids", -FRAME_BUFFER_LENGTH, -1)

                    safe_redis_operation(redis_client.rpush, "force", json.dumps(forces_frame.tolist()))
                    safe_redis_operation(redis_client.ltrim, "force",  -FRAME_BUFFER_LENGTH, -1)
                    safe_redis_operation(redis_client.rpush, "mks", json.dumps(markers_frame.tolist()))
                    safe_redis_operation(redis_client.ltrim, "mks",  -FRAME_BUFFER_LENGTH, -1)
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
        self.processed_frame_ids = deque(maxlen=2*FRAME_BUFFER_LENGTH)
        self.processed_cycles = deque(maxlen=2*CYCLE_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"
        self.cycle_counter = 0  # For the detection of cycles
        self.cycle_idx = 0  # For the treatment of the cycles
        self.cycle_start_id = None

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
        # print("Identifying cycle start...")

        force_filtered = self.data_filter(forces_all[0, 0:3, :], 2, MARKER_FREQUENCY, 10)
        current_cycle_idx = np.ones((forces_all[0].shape[1], )) * self.cycle_counter

        right_foot_on_ground_idx = force_filtered[2, :] > FORCE_MIN_THRESHOLD*2 * MASS
        right_foot_on_ground_idx = np.astype(right_foot_on_ground_idx, int)
        heel_strike_idx = np.where(np.diff(right_foot_on_ground_idx) == 1)[0] + 1
        # toe_off_idx = np.where(np.diff(right_foot_on_ground_idx) == -1)[0] + 1

        # # Identification : OK
        # plt.figure()
        # plt.plot(force_filtered[2, :])
        # for frame in heel_strike_idx:
        #     plt.axvline(x=frame, color='red', linestyle='--', label='Heel Strike')
        # for frame in toe_off_idx:
        #     plt.axvline(x=frame, color='green', linestyle='--', label='Toe Off')
        # plt.savefig("cycle_identification.png")
        # # plt.show()

        for i_cycle in range(heel_strike_idx.shape[0]):
            self.cycle_counter += 1
            if heel_strike_idx.shape[0] > i_cycle + 1:
                current_cycle_idx[heel_strike_idx[i_cycle]:heel_strike_idx[i_cycle+1]] = self.cycle_counter
            else:
                current_cycle_idx[heel_strike_idx[i_cycle]:] = self.cycle_counter

            # # TODO: remove ?
            # safe_redis_operation(redis_client.rpush, "current_cycle_idx", json.dumps(current_cycle_idx.tolist()))
            # safe_redis_operation(redis_client.ltrim, "current_cycle_idx", -FRAME_BUFFER_LENGTH, -1)

        return heel_strike_idx


    def process(self):
        try:
            global MODEL

            new_indices, new_frame_ids, all_frame_ids = get_new_indices(self.processed_frame_ids, print_option=False)

            if len(new_frame_ids) > 99:
                print([new_frame_ids[0], len(new_frame_ids), new_frame_ids[-1]])
                forces_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)
                mks_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("mks", 0, -1)]
                mks_all = np.array(mks_all).transpose(1, 2, 0)
                mks_name = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("mks_name", 0, -1)]

                if mks_all.shape[2] != len(all_frame_ids) or forces_all.shape[2] != len(all_frame_ids):
                    # logging.info("Les données de mks et forces ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    return

                forces = forces_all[:, :, new_indices]
                heel_strike_idx = self.identify_cycle_start(forces)

                if heel_strike_idx.shape[0] > 0:
                    if self.cycle_start_id is None:
                        # We skip on purpose everything before the first heel strike is detected
                        self.cycle_start_id = str(new_frame_ids[heel_strike_idx[0]])
                        print("initiqlization : start id = ", self.cycle_start_id)
                        self.processed_frame_ids.extend(new_frame_ids[:heel_strike_idx[0]])
                    else:
                        cycle_stop_id = str(new_frame_ids[heel_strike_idx[0]])

                        # Récupérer les données pour ce cycle uniquement
                        cycle_start_idx = all_frame_ids.index(self.cycle_start_id)
                        cycle_stop_idx = all_frame_ids.index(cycle_stop_id)

                        if cycle_stop_idx - cycle_start_idx < 30:
                            if len(heel_strike_idx) > 1:
                                cycle_stop_id = str(new_frame_ids[heel_strike_idx[1]])
                                cycle_stop_idx = all_frame_ids.index(cycle_stop_id)
                            else:
                                # logging.info("Cycle trop court, pas de traitement.")
                                return

                        print("start : ", cycle_start_idx, " / stop : ", cycle_stop_idx)
                        print("cycle idx : ", self.cycle_idx)

                        mks = mks_all[:, :, cycle_start_idx:cycle_stop_idx]
                        forces = forces_all[:, :, cycle_start_idx:cycle_stop_idx]

                        if MODEL is not None:
                            print("Calcul IK/ID...")

                            # # Check that all frames have the same markers
                            # mks_names_this_cycle = mks_name[cycle_start_idx]
                            # for frame in range(cycle_start_idx, cycle_stop_idx):
                            #     if mks_name[frame] != mks_names_this_cycle:
                            #         logging.error("Les noms des marqueurs ne correspondent pas pour tous les frames.")
                            #         return

                            q, qdot, qddot = self.calculate_ik(MODEL, mks, mks_name)
                            tau = self.calculate_id(MODEL, forces, q, qdot, qddot)

                            print("q envoyé: ", q.shape)

                            # Stocker les résultats dans Redis
                            # Stocker l'indice dans une liste séparée pour suivre l'ordre
                            safe_redis_operation(redis_client.rpush, "cycle_idx", self.cycle_idx)
                            safe_redis_operation(redis_client.ltrim, "cycle_idx", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "q", json.dumps(q.tolist()))
                            safe_redis_operation(redis_client.ltrim, "q",  -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "qdot", json.dumps(qdot.tolist()))
                            safe_redis_operation(redis_client.ltrim, "qdot",  -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "qddot", json.dumps(qddot.tolist()))
                            safe_redis_operation(redis_client.ltrim, "ddotq",  -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "tau", json.dumps(tau.tolist()))
                            safe_redis_operation(redis_client.ltrim, "tau",  -CYCLE_BUFFER_LENGTH, -1)

                        self.cycle_start_id = cycle_stop_id
                        self.processed_frame_ids.extend(all_frame_ids[cycle_start_idx: cycle_stop_idx])
                        self.cycle_idx += 1


        except Exception as e:
            logging.error(f"Erreur lors du traitement des données: {e}")

    def calculate_ik(self, model: biorbd.Model, mks, labels):
        try:
            n_frames = mks.shape[2]
            marker_names = tuple(n.to_string() for n in MODEL.technicalMarkerNames())
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

    def calculate_id(self, model: biorbd.Model, force, q, qdot, qddot):
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


class BayesianOptimizer:
    """Traite les données pour determiner quels parametres de stimulation essayer"""

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
        # self.processed_frame_ids = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        # self.processed_cycles = deque(maxlen=2 * CYCLE_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"
        # self.cycle_counter = 0
        self.current_cycle = None

        # Optimization parameters
        # Define the variable bounds
        self.bounds = [
            Real(20, 50, name="R_frequency"),  # Hz
            Real(8, 20, name="R_intensity"),  # mA
            Real(200, 500, name="R_width"),  # micros
            Real(20, 50, name="L_frequency"),  # Hz
            Real(8, 20, name="L_intensity"),  # mA
            Real(200, 500, name="L_width"),  # micros
        ]

        # Define the objective weightings
        # TODO: Charbie -> how do we chose which objectives to minimize ?
        self.weight_comddot = 1
        self.weight_angular_momentum = 1
        self.weight_enegy = 1
        self.weight_ankle_power = -1


    def start_optimizing(self):
        self.running = True

        while self.running:
            try:
                if is_redis_connected():
                    """Perform Bayesian optimization using Gaussian Processes."""

                    # gp_minimize will try to find the minimal value of the objective function.
                    result = gp_minimize(
                        func=lambda params: self.make_an_iteration(params),
                        dimensions=self.bounds,
                        n_calls=100,  # number of evaluations of f
                        acq_func="LCB",  # "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
                        kappa=5,  # *
                        random_state=0,  # *
                        n_jobs=1,
                    )  # x0, y0, kappa[exploitation, exploration], xi [minimal improvement default 0.01]

                    # TODO: stop when the same point has been hit t time (t=5 in general)

                    optimal_parameter_values = result.x

                    # TODO: save the optimal parameters

                    # TODO: Plot the optimal values

            except Exception as e:
                logging.error(f"Erreur dans BayesianOptimizer: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.wait()

    def process(self):
        try:

            global MODEL

            cycle_indices = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("cycle_idx", 0, -1)]
            q_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("q", 0, -1)]
            qdot_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("qdot", 0, -1)]
            qddot_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("qddot", 0, -1)]
            tau_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("tau", 0, -1)]

            if len(q_all) != len(cycle_indices) or len(qdot_all) != len(cycle_indices) or len(qddot_all) != len(cycle_indices) or len(tau_all) != len(cycle_indices):
                # We are in a weird state, it is better to wait for the next loop
                return

            if len(q_all) > 0 and len(cycle_indices) > 0:
                if self.current_cycle is None:
                    self.current_cycle = cycle_indices[0]
                elif self.current_cycle < cycle_indices[-1]:
                    self.current_cycle += 1

                q = np.array(q_all[cycle_indices.index(self.current_cycle)])
                print("q recu : ", q.shape)

            # TODO: if bounds are updated, we need to update the bounds in the optimizer




        except Exception as e:
            logging.error(f"Erreur lors loptimisation bayesienne: {e}")


    def compute_mean_cycle(self, cycles):

        global MARKER_FREQUENCY

        nb_frames = [cycles["nb_frames"][-i_cycle] for i_cycle in range(10)]
        nb_interpolation_frames = np.mean(np.array(nb_frames))
        q_interpolated = np.zeros((NB_DOF, nb_interpolation_frames, 10))
        qdot_interpolated = np.zeros((NB_DOF, nb_interpolation_frames, 10))
        qddot_interpolated = np.zeros((NB_DOF, nb_interpolation_frames, 10))
        tau_interpolated = np.zeros((NB_DOF, nb_interpolation_frames, 10))
        for i_cycle in range(10):
            current_q = cycles["q"][-i_cycle]
            current_qdot = cycles["qdot"][-i_cycle]
            current_qddot = cycles["qddot"][-i_cycle]
            current_tau = cycles["tau"][-i_cycle]
            current_nb_frames = len(current_q)
            time_vector = np.linspace(0, (current_nb_frames - 1) * 1 / MARKER_FREQUENCY, current_nb_frames)
            time_vector_interpolated = np.linspace(
                0, (current_nb_frames - 1) * 1 / MARKER_FREQUENCY, nb_interpolation_frames
            )

            interp_func_q = interp1d(current_q, time_vector, kind="cubic")
            interp_func_qdot = interp1d(current_qdot, time_vector, kind="cubic")
            interp_func_qddot = interp1d(current_qddot, time_vector, kind="cubic")
            interp_func_tau = interp1d(current_tau, time_vector, kind="cubic")

            q_interpolated[:, :, i_cycle] = interp_func_q(time_vector_interpolated)
            qdot_interpolated[:, :, i_cycle] = interp_func_qdot(time_vector_interpolated)
            qddot_interpolated[:, :, i_cycle] = interp_func_qddot(time_vector_interpolated)
            tau_interpolated[:, :, i_cycle] = interp_func_tau(time_vector_interpolated)

        q_mean = np.mean(q_interpolated, axis=2)
        qdot_mean = np.mean(qdot_interpolated, axis=2)
        qddot_mean = np.mean(qddot_interpolated, axis=2)
        tau_mean = np.mean(tau_interpolated, axis=2)

        return q_mean, qdot_mean, qddot_mean, tau_mean

    def make_an_iteration(self, params):

        # Current values of the optimized FES parameters
        R_frequency = params[0]
        R_intensity = params[1]
        R_width = params[2]
        L_frequency = params[3]
        L_intensity = params[4]
        L_width = params[5]

        # Send FES with these parameter values
        # Setup Right leg
        self.visualization_widget.set_chanel_inputs(
            channel=0,
            channel_layout=None,
            name_input="Right",
            amplitude_input=R_intensity,  # TODO: @ophelielariviere, is it the right parameter ?
            pulse_width_input=R_width,
            frequency_input=R_frequency,
            mode_input=Modes.SINGLE,
        )
        # Setup Left leg
        self.visualization_widget.set_chanel_inputs(
            channel=1,
            channel_layout=None,
            name_input="Left",
            amplitude_input=L_intensity,  # TODO: @ophelielariviere, is it the right parameter ?
            pulse_width_input=L_width,
            frequency_input=L_frequency,
            mode_input=Modes.SINGLE,
        )
        # Stimulate
        self.visualization_widget.start_stimulation(self, [0, 1])

        # Collect data while waiting for the subject to get a stable walking pattern with these parameters
        cycles = {
            # "StanceDuration_L": [],
            # "StanceDuration_R": [],
            # "Cycleduration": [],
            # "StepWidth": [],
            # "StepLength_L": [],
            # "StepLength_R": [],
            # "PropulsionDuration_L": [],
            # "PropulsionDuration_R": [],
            # "Cadence": [],
            "q": [],
            "qdot": [],
            "qddot": [],
            "tau": [],
            "nb_frames": [],
        }
        old_cycledata = None
        new_cycledata = "Fake_data"
        stable = False
        while not stable:
            while old_cycledata == new_cycledata:  # No new data to collect
                new_cycledata = self.visualization_widget.buffer.get()
                time.sleep(0.1)
            # Add the new cycle data to the list of cycles
            new_gait_parameters = new_cycledata["gait_parameter"]
            new_q = new_cycledata["Angle"]
            new_qdot = new_cycledata["VitAng"]
            new_qddot = new_cycledata["AccAng"]
            new_tau = new_cycledata["Tau"]
            cycles["StanceDuration_L"] += new_gait_parameters["StanceDuration_L"]
            cycles["StanceDuration_R"] += new_gait_parameters["StanceDuration_R"]
            cycles["Cycleduration"] += new_gait_parameters["Cycleduration"]
            cycles["StepWidth"] += new_gait_parameters["StepWidth"]
            cycles["StepLength_L"] += new_gait_parameters["StepLength_L"]
            cycles["StepLength_R"] += new_gait_parameters["StepLength_R"]
            cycles["PropulsionDuration_L"] += new_gait_parameters["PropulsionDuration_L"]
            cycles["PropulsionDuration_R"] += new_gait_parameters["PropulsionDuration_R"]
            cycles["Cadence"] += new_gait_parameters["Cadence"]
            cycles["q"] += new_q
            cycles["qdot"] += new_qdot
            cycles["qddot"] += new_qddot
            cycles["tau"] += new_tau
            cycles["nb_frames"] += new_q.shape[1]
            if len(cycles["Cycleduration"]) > 10:
                # Compute the std of the last 10 cycles
                StanceDuration_L_std = np.nanstd(cycles["StanceDuration_L"][-10:])
                StanceDuration_R_std = np.nanstd(cycles["StanceDuration_R"][-10:])
                Cycleduration_std = np.nanstd(cycles["Cycleduration"][-10:])
                StepWidth_std = np.nanstd(cycles["StepWidth"][-10:])
                StepLength_L_std = np.nanstd(cycles["StepLength_L"][-10:])
                StepLength_R_std = np.nanstd(cycles["StepLength_R"][-10:])
                PropulsionDuration_L_std = np.nanstd(cycles["PropulsionDuration_L"][-10:])
                PropulsionDuration_R_std = np.nanstd(cycles["PropulsionDuration_R"][-10:])
                Cadence_std = np.nanstd(cycles["Cadence"][-10:])

                # Check if the last 10 cycles are stable
                stable = (
                        StanceDuration_L_std < 0.05  # 5% of the cycle
                        and StanceDuration_R_std < 0.05  # 5% of the cycle
                        and Cycleduration_std < 0.05  # 5% of the cycle
                        and StepWidth_std < 0.05  # 5cm
                        and StepLength_L_std < 0.05  # 5cm
                        and StepLength_R_std < 0.05  # 5cm
                        and PropulsionDuration_L_std < 0.05  # 5% of the cycle
                        and PropulsionDuration_R_std < 0.05  # 5% of the cycle
                        and Cadence_std < 5
                )
            old_cycledata = new_cycledata

        # Stop the stimulation
        self.visualization_widget.pause_stimulation()

        # Compute the mean cycle
        q_mean, qdot_mean, qddot_mean, tau_mean = self.compute_mean_cycle(cycles)

        # Compute objective values
        objective_value = self.objective(self.model, q_mean, qdot_mean, qddot_mean, tau_mean, R_intensity,
                                         L_intensity)

        return objective_value

    @staticmethod
    def compute_com_acceleration(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray):

        nb_frames = q.shape[1]

        comddot = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            comddot[i_frame] = np.linalg.norm(
                model.CoMddot(q[:, i_frame], qdot[:, i_frame], qddot[:, i_frame]).to_array()
            )

        return comddot

    @staticmethod
    def compute_angular_momentum(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray):

        nb_frames = q.shape[1]

        angular_momentum = np.zeros((nb_frames,))
        for i_frame in range(nb_frames):
            angular_momentum[i_frame] = np.linalg.norm(
                model.angularMomentum(q[:, i_frame], qdot[:, i_frame]).to_array()
            )

        return angular_momentum

    @staticmethod
    def compute_energy(qdot, tau, R_intensity, L_intensity, time_vector):
        """
        Since the time is the same, min energy and power gives the same thing (same min).
        """

        voltage = 30  # TODO: @ophelielariviere, what is the voltage ?
        power_stim = np.abs(R_intensity * voltage) + np.abs(L_intensity * voltage)
        power_total = np.sum(np.abs(tau * qdot), axis=0)
        power_human = power_total - power_stim
        energy_human = np.trapezoid(power_human, x=time_vector)

        return energy_human

    def compute_ankle_power(self, qdot, tau, time_vector):
        # TODO: find which idx is the flexion [0, 1, 2]
        ankle_index = [self.dof_corr["RAnkle"][0], self.dof_corr["LAnkle"][0]]
        sum_ankles = np.sum(np.abs(tau[ankle_index, :] * qdot[ankle_index, :]), axis=0)
        return np.trapezoid(sum_ankles, x=time_vector)


    def objective(self, model, q, qdot, qddot, tau, R_intensity, L_intensity):

        nb_frames = q.shape[1]
        read_frequency = 100  # Hz  # TODO: @ophelielariviere, is it always 100 Hz ?
        time_vector = np.linspace(0, (nb_frames - 1) * 1 / read_frequency, nb_frames)

        comddot = self.compute_com_acceleration(model, q, qdot, qddot)
        angular_momentum = self.compute_angular_momentum(model, q, qdot, qddot)
        energy_human = self.compute_energy(qdot, tau, R_intensity, L_intensity, time_vector)
        power_ankle = self.compute_ankle_power(self, qdot, tau, time_vector)

        return (
                self.weight_comddot * comddot
                + self.weight_angular_momentum * angular_momentum
                + self.weight_enegy * energy_human
                + self.weight_ankle_power * power_ankle
        )

    def save_optimal_bayesian_parameters(self, result):
        """
        result contains:
            - fun [float]: function value at the minimum.
            - models: surrogate models used for each iteration.
            - x_iters [list of lists]: location of function evaluation for each iteration.
            - func_vals [array]: function value for each iteration.
            - space [Space]: the optimization space.
            - specs [dict]`: the call specifications.
            - rng [RandomState instance]: State of the random state at the end of minimization.
        """
        # TODO

        save_file_name = self.visualization_widget.path_to_saveData + "optimal_bayesian_parameters.txt"
        with open(save_file_name, "w") as f:
            f.write("Optimal parameters found through Bayesian optimization : \n\n")
            f.write("Frequency right = %.4f\n" % result.x[0])
            f.write("Intensity right = %.4f\n" % result.x[1])
            f.write("Width right = %.4f\n" % result.x[2])
            f.write("Frequency left = %.4f\n" % result.x[3])
            f.write("Intensity left = %.4f\n" % result.x[4])
            f.write("Width left = %.4f\n" % result.x[5])
            f.write("\nOptimal cost function value = %.4f\n" % result.fun)
        return

    def plot_bayesian_optim_results(self, result):
        # TODO

        print("Best found minimum:")
        print("X = %.4f, Y = %.4f" % (result.x[0], result.x[1]))
        print("f(x,y) = %.4f" % result.fun)

        # Optionally, plot convergence
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132, projection="3d")
        ax2 = fig.add_subplot(133, projection="3d")

        # Convergence plot
        ax0.plot(result.func_vals, marker="o")
        ax0.set_title("Convergence Plot")
        ax0.set_xlabel("Number of calls")
        ax0.set_ylabel("Objective function value")

        # Plot the function sampling on the right side
        x_iters_array = np.array(result.x_iters)
        func_vals_array = np.array(result.func_vals)
        colors_min = np.min(func_vals_array)
        colors_max = np.max(func_vals_array)
        normalized_cmap = (func_vals_array - colors_min) / (colors_max - colors_min)
        colors = cm["viridis"](normalized_cmap)
        p1 = ax1.scatter(x_iters_array[:, 0], x_iters_array[:, 1], x_iters_array[:, 2], c=colors, marker=".")
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Intensity")
        ax1.set_zlabel("Width")
        ax1.set_title("Function sampling Right")

        # Plot the function sampling on the left side
        p2 = ax2.scatter(x_iters_array[:, 3], x_iters_array[:, 4], x_iters_array[:, 5], c=colors, marker=".")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Intensity")
        ax2.set_zlabel("Width")
        ax2.set_title("Function sampling Left")

        cbar = fig.colorbar(p1)
        cbar.set_label("Objective function value")
        plt.show()



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
        self.processed_frame_ids = deque(maxlen=2*FRAME_BUFFER_LENGTH)
        self.data_received = "Not initialized"

    def start_processing(self):
        global ACTIVATE_STIMULATOR, START_STIMULATION, STOP_STIMULATOR

        while self.running:
            try:
                if is_redis_connected():

                    if ACTIVATE_STIMULATOR:
                        self.activate_stimulator()
                        ACTIVATE_STIMULATOR = False

                    if START_STIMULATION:
                        self.call_start_stimulation(self.last_channels)
                        START_STIMULATION = False

                    if STOP_STIMULATOR:
                        self.stop_stimulator()
                        STOP_STIMULATOR = False

                    self.stimulation_process()
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Erreur dans StimulationProcessor: {e}")
                time.sleep(0.01)

    def stimulation_process(self):
        try:
            new_indices, new_frame_ids, all_frame_ids = get_new_indices(self.processed_frame_ids, print_option=False)

            if len(new_indices) > 0:
                forces_all = [json.loads(x.decode('utf-8')) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)

                if forces_all.shape[2] != len(all_frame_ids):
                    # logging.info("Les données de forces ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    return

                force_data = forces_all[:, :, new_indices]
                self.processed_frame_ids.extend(new_frame_ids)

                # Was like this before:
                # fyr = force_data[0][2]  # Force Y droite
                # fzr = force_data[0][5]  # Force Z droite
                # fyl = force_data[0][8]  # Force Y gauche
                # fzl = force_data[0][11]  # Force Z gauche
                fyr = force_data[0][4]  # Force Y droite
                fzr = force_data[0][5]  # Force Z droite
                fyl = force_data[1][4]  # Force Y gauche
                fzl = force_data[1][5]  # Force Z gauche
                if len(fyr) > 20 and MASS:
                    info_feet = {
                        "right": self.detect_phase_force(fyr, fzr, fzl, 1, MASS),
                        "left": self.detect_phase_force(fyl, fzl, fzr, 2, MASS),
                    }
                    self.manage_stimulation(info_feet)

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

    def manage_stimulation(self, info_feet):
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
                    self.call_start_stimulation(new_channels)
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

    def call_start_stimulation(self, channel_to_send):
        try:
            if not self.stimulator_is_active:
                self.activate_stimulator()
                time.sleep(1)  # Attendre que le stimulateur soit prêt

            if self.stimulator_is_sending_stim:
                self.call_pause_stimulation()

            stim_params = safe_redis_operation(redis_client.lrange, "stimulation_parameters", 0, -1)
            if stim_params:
                stimulator_parameters = json.loads(stim_params[-1])

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
        self.process_idik = False
        self.do_look_need_send_stim = False
        self.stimulation_mode = StimulationMode.MANUAL
        self.channel_bounds = {f"Canal {i}": DEFAULT_BOUNDS for i in range(1, 9)}
        self.discomfort = 0
        self.DataToPlot = self.initialize_data_to_plot()

        # Initialize UI components
        self.init_ui()

        # # Timer pour mettre à jour les graphes toutes les secondes
        # self.graph_update_timer = QTimer(self)
        # self.graph_update_timer.timeout.connect(self.update_data_and_graphs)
        # self.graph_update_timer.start(100)


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

        # Mode de stimulation
        main_layout.addWidget(self.create_optimization_mode())

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
        global MODEL_FILE_NAME, MODEL

        if not is_redis_connected():
            QMessageBox.warning(self, "Erreur", "La connexion Redis n'est pas établie. Veuillez patienter...")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier")
        if file_name:
            try:
                MODEL_FILE_NAME = file_name
                MODEL = biorbd.Model(file_name)
                logging.info(f"Fichier modèle chargé: {file_name}")
                self.model_label.setText(file_name.split('/')[-1])
            except Exception as e:
                logging.error(f"Erreur lors du chargement du fichier: {str(e)}")
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier: {str(e)}")

    def update_mass(self, mass_value):
        """Met à jour la masse du participant"""
        if not is_redis_connected():
            QMessageBox.warning(self, "Erreur", "La connexion Redis n'est pas établie. Veuillez patienter...")
            return

        try:
            MASS = float(mass_value)
            logging.info(f"Masse mise à jour: {MASS} kg")
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
                amplitude_input.setRange(DEFAULT_BOUNDS["Amplitude"][0], DEFAULT_BOUNDS["Amplitude"][1])
                amplitude_input.setSuffix(" mA")
                pulse_width_input = QSpinBox()
                pulse_width_input.setRange(DEFAULT_BOUNDS["Pulse Width"][0], DEFAULT_BOUNDS["Pulse Width"][1])
                pulse_width_input.setSuffix(" µs")
                frequency_input = QSpinBox()
                frequency_input.setRange(DEFAULT_BOUNDS["Frequency"][0], DEFAULT_BOUNDS["Frequency"][1])
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

                # Enregistrer les widgets pour le canal sélectionné
                self.set_channel_inputs(
                    channel,
                    channel_layout,
                    name_input,
                    amplitude_input,
                    pulse_width_input,
                    frequency_input,
                    mode_input,
                )

    @staticmethod
    def activate_stimulator():
        global ACTIVATE_STIMULATOR
        ACTIVATE_STIMULATOR=True

    def old_activate_stimulator(self):
        self.channels = []
        for channel, inputs in self.channel_inputs.items():
            channel_obj = Channel(
                no_channel=channel,
                name=inputs["name_input"].text(),
                amplitude=inputs["amplitude_input"].value(),
                pulse_width=inputs["pulse_width_input"].value(),
                frequency=inputs["frequency_input"].value(),
                mode=Modes.SINGLE,  # inputs["mode_input"].currentText(),
                device_type=Device.Rehastimp24,
            )

            self.channels.append(channel_obj)
        if self.channels:
            self.stimulator.init_stimulation(list_channels=self.channels)

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def update_stimulation(self):
        if self.stimulator is not None:
            self.stimulator.update_stimulation()

    def manual_optim_chosen(self):
        self.update_button.setEnabled(True)
        self.start_bayesian_optim_button.setEnabled(False)
        self.stop_bayesian_optim_button.setEnabled(False)
        # TODO: Charbie -> add the ICL buttons

    def bayesian_optim_chosen(self):
        self.update_button.setEnabled(False)
        self.start_bayesian_optim_button.setEnabled(True)
        self.stop_bayesian_optim_button.setEnabled(True)
        # TODO: Charbie -> add the ICL buttons

    def ilc_optim_chosen(self):
        self.update_button.setEnabled(False)
        self.start_bayesian_optim_button.setEnabled(False)
        self.stop_bayesian_optim_button.setEnabled(False)
        # TODO: Charbie -> add the ICL buttons


    def start_stimulation(self):
        global START_STIMULATION
        START_STIMULATION=True

        if self.is_manual_mode:
            self.update_button.setEnabled(True)
        elif self.is_bayesian_mode:
            self.start_bayesian_optim_button.setEnabled(True)
            self.stop_bayesian_optim_button.setEnabled(True)


    def stop_stimulator(self):
        global STOP_STIMULATOR
        STOP_STIMULATOR=True

    def create_stimulation_controls(self):
        """Crée les contrôles de stimulation"""
        layout = QHBoxLayout()

        self.activate_button = QPushButton("Activer Stimulateur")
        self.activate_button.clicked.connect(self.activate_stimulator)

        self.update_button = QPushButton("Actualiser Paramètres")
        self.update_button.clicked.connect(self.update_stimulation_parameter)

        self.start_button = QPushButton("Envoyer Stimulation")
        self.start_button.clicked.connect(self.start_stimulation)

        self.stop_button = QPushButton("Arrêter Stimulateur")
        self.stop_button.clicked.connect(self.stop_stimulator)

        self.checkpauseStim = QCheckBox("Stop tying send stim")
        self.checkpauseStim.setChecked(True)
        self.checkpauseStim.stateChanged.connect(self.pause_fonction_to_send_stim)

        layout.addWidget(self.checkpauseStim)
        layout.addWidget(self.activate_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.update_button)
        layout.addWidget(self.stop_button)

        return layout

    def create_optimization_mode(self):
        """Créer les boutons pour choisir si la stimulation est en mode manuel ou optimisé."""
        groupbox = QGroupBox("Stimulation Parameter Mode:")
        layout = QGridLayout()

        # Manual Mode
        self.manual_mode_button = QRadioButton("Manual", self)
        self.manual_mode_button.setChecked(True)
        self.manual_mode_button.toggled.connect(self.manual_optim_chosen)
        self.update_button = QPushButton("Actualiser Paramètre Stim")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self.update_stimulation)

        layout.addWidget(self.manual_mode_button, 0, 0, 1, 1)
        layout.addWidget(self.update_button, 0, 1, 1, 1)

        # Bayesian Optimization Mode
        self.bayesian_mode_button = QRadioButton("Bayesian Optimization", self)
        self.bayesian_mode_button.toggled.connect(self.bayesian_optim_chosen)
        self.start_bayesian_optim_button = QPushButton("Start Optim")
        self.start_bayesian_optim_button.setEnabled(False)
        self.start_bayesian_optim_button.clicked.connect(self.start_bayesian_optimization)
        self.stop_bayesian_optim_button = QPushButton("Early Termination Optim")
        self.stop_bayesian_optim_button.setEnabled(False)
        self.stop_bayesian_optim_button.clicked.connect(self.stop_bayesian_optimization)

        layout.addWidget(self.bayesian_mode_button, 1, 0, 1, 1)
        layout.addWidget(self.start_bayesian_optim_button, 1, 1, 1, 1)
        layout.addWidget(self.stop_bayesian_optim_button, 1, 2, 1, 1)

        # Iterative Learning Control Mode
        self.ilc_mode_button = QRadioButton("Iterative Learning Control", self)
        self.ilc_mode_button.toggled.connect(self.ilc_optim_chosen)
        self.ilc_mode_button.setEnabled(False)  # TODO: Charbie -> Implement ILC, for now always disabled
        layout.addWidget(self.ilc_mode_button, 2, 0, 1, 1)

        # Channel Bounds Section
        self.channel_bounds_inputs = {f"Canal {i}": {} for i in range(1, 9)}
        for i in range(1, 9):
            channel_label = QLabel(f"Canal {i} :")
            layout.addWidget(channel_label, 3, i - 1)

            for parameter_name in DEFAULT_BOUNDS.keys():
                if parameter_name not in self.channel_bounds_inputs:
                    self.channel_bounds_inputs[parameter_name] = {}

                channel_min_bound = QSpinBox()
                channel_min_bound.setRange(DEFAULT_BOUNDS[parameter_name][0], DEFAULT_BOUNDS[parameter_name][1])
                channel_min_bound.setValue(DEFAULT_BOUNDS[parameter_name][0])
                channel_max_bound = QSpinBox()
                channel_max_bound.setRange(DEFAULT_BOUNDS[parameter_name][0], DEFAULT_BOUNDS[parameter_name][1])
                channel_max_bound.setValue(DEFAULT_BOUNDS[parameter_name][1])

                self.channel_bounds_inputs[f"Canal {i}"][parameter_name] = [channel_min_bound, channel_max_bound]
                layout.addWidget(channel_min_bound, 4, i - 1, 1, 1)
                layout.addWidget(channel_max_bound, 5, i - 1, 1, 1)

        stable_cycles_label = QLabel(f"There were <b>{self.num_stable_cycles}</b> stable cycles")
        layout.addWidget(stable_cycles_label, 0, 5)
        current_cost_label = QLabel(f"The current cost is <b>{self.current_cost}</b>")
        layout.addWidget(current_cost_label, 1, 5)
        discomfort_label = QLabel(f"Discomfort  :")
        layout.addWidget(discomfort_label, 2, 4)
        discomfort_box = QSpinBox()
        discomfort_box.setRange(0, 10)
        discomfort_box.setValue(self.discomfort)
        layout.addWidget(discomfort_box, 2, 5)
        discomfort_button = QPushButton("Set discomfort")
        discomfort_button.clicked.connect(lambda: setattr(self, "discomfort", discomfort_box.value()))
        layout.addWidget(discomfort_button, 2, 6)

        # Set the groupbox layout
        groupbox.setLayout(layout)
        return groupbox

    def set_channel_inputs(
        self, channel, channel_layout, name_input, amplitude_input, pulse_width_input, frequency_input, mode_input
    ):
        # Enregistrer les widgets pour le canal sélectionné
        self.channel_inputs[channel] = {
            "layout": channel_layout,
            "name_input": name_input,
            "amplitude_input": amplitude_input,
            "pulse_width_input": pulse_width_input,
            "frequency_input": frequency_input,
            "mode_input": mode_input,
        }

    def pause_fonction_to_send_stim(self):
        """Met à jour l'état d'envoi de stimulation"""
        self.do_look_need_send_stim = not self.checkpauseStim.isChecked()

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
                safe_redis_operation(redis_client.ltrim, "stimulation_parameters",  -FRAME_BUFFER_LENGTH, -1)
                logging.info("Paramètres de stimulation mis à jour")
            except Exception as e:
                logging.error(f"Erreur lors de la mise à jour des paramètres: {e}")

    def start_bayesian_optimization(self):
        """Démarre l'optimisation Bayésienne."""
        self.bayesian_optimizer = BayesianOptimizer(self)
        result = self.bayesian_optimizer.perform_bayesian_optim()
        self.save_optimal_bayesian_parameters(result)
        self.bayesian_optimizer.plot_bayesian_optim_results(result)
        # TODO : Charbie -> stimulate with these parameters for a few minutes ?

    def stop_bayesian_optimization(self):
        """Arrête l'optimisation Bayésienne."""
        # TODO save the best parameters
        pass
    
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

    # Redis manager
    redis_manager = RedisConnectionManager()

    # Data receiver (goal: interaction with Qualisys)
    data_receiver = DataReceiver("127.0.0.1", 50000)

    # Data processor (goal: ID, IK)
    data_processor = DataProcessor()

    # Stimulation processor (goal: determine if a stim is needed + interaction with stimulator)
    stimulation_processor = StimulationProcessor()

    # Bayesian optimizer (goal: determine which stimulation parameters to try)
    bayesian_optimizer = BayesianOptimizer()

    # --- Thread activation --- #
    threading.Thread(target=redis_manager.run, daemon=False).start()
    threading.Thread(target=data_receiver.start_receiving, daemon=False).start()
    threading.Thread(target=data_processor.start_processing, daemon=False).start()
    threading.Thread(target=stimulation_processor.start_processing, daemon=False).start()
    threading.Thread(target=bayesian_optimizer.start_optimizing, daemon=False).start()

    # Start the GUI
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()