"""
Before starting the code, start Docker Desktop or run in terminal
`docker run --name redis-server -p 6379:6379 -d redis`

Info:
The Redis database contains
1. Data at each frame (force, mks, mks_names, frame_ids)
2. Data at each cycle (q, tau, cycle_ids)

Note: Always start the interface (this code) before the data server (code on the other computer).
"""

import os
import  pickle
import datetime
import sys
from enum import Enum
import logging
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QGridLayout,
    QRadioButton,
)
from PyQt5.QtCore import QTimer
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
from collections import deque
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import threading
from skopt import gp_minimize
from skopt.space import Real


# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)

# Constantes globales
FRAME_BUFFER_LENGTH = 800
CYCLE_BUFFER_LENGTH = 100
REDIS_HOST = "localhost"
REDIS_PORT = 6379

MARKER_FREQUENCY = 100
FORCE_MIN_THRESHOLD = 0.1

# Flags to check for stimulation and processing
ACTIVATE_STIMULATOR = False
START_STIMULATION = False
STOP_STIMULATOR = False
PROCESS_ID_IK = False
RUN_OPTIMISATION = False

# Single value shared variables (instead of duplicating the information in the redis database)
MASS = 66  # Initial value only (will be set by Interface.update_mass)
MODEL_FILE_NAME = "C:/Users/olarivie/PycharmProjects/eWalking_FES_Interface/example/ECH.bioMod"  # Will be set by Interface.upload_file
MODEL = biorbd.Model(MODEL_FILE_NAME)  # Will be set by Interface.upload_file
DOF_CORR = {
    "Pelvis": [3, 4, 5],
    "RHip": [6, 7],
    "RKnee": [8],
    "RAnkle": [9],
    "LHip": [10, 11],
    "LKnee": [12],
    "LAnkle": [13],
}
NB_DOF = MODEL.nbQ()
DEFAULT_BOUNDS = {
    "Amplitude": [0, 100],  # Amplitude en mA
    "Pulse Width": [0, 1000],  # Largeur d'impulsion en microsecondes
    "Frequency": [0, 200],  # Fréquence en Hz
}
DISCOMFORT = 0

# Instance Redis globale
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
redis_client.flushdb()
IS_REDIS_CONNECTED = True


class StimulationMode(Enum):
    MANUAL = "manual"
    BAYESIAN = "bayesian"
    ILC = "ilc"
    # TODO: implement ILC based on https://www.sciencedirect.com/science/article/abs/pii/S0967066120300046


def safe_redis_operation(operation, *args, **kwargs):
    global IS_REDIS_CONNECTED
    if not IS_REDIS_CONNECTED:
        logging.warning("Redis n'est pas connecté")
        return None
    try:
        return operation(*args, **kwargs)
    except redis.RedisError as e:
        logging.error(f"Erreur Redis: {str(e)}")
        return None


def get_new_indices(timestamps, processed_frame_timestamps, print_option=False):
    """Filtrer pour ne garder que les nouveaux timestamps"""
    global redis_client

    try:
        new_indices = [i for i, time in enumerate(timestamps) if time not in processed_frame_timestamps]
        new_timestamps = [time for time in timestamps if time not in processed_frame_timestamps]
        new_timestamps = np.array(new_timestamps)
        new_indices = np.array(new_indices)

        if print_option:
            if len(processed_frame_timestamps) > 0:
                print("processed ", processed_frame_timestamps[-1])
                print("timestamps ", timestamps[0], timestamps[-1])
                print("new indices ", new_indices[0], new_indices[-1])
    except:
        logging.error("erreur lors de l'identification des new indices.")

    return new_indices, new_timestamps

def get_indices_of_these_timestamps(target_timestamps, all_timestamps):
    indices = []
    for timestamp in target_timestamps:
        if timestamp in all_timestamps:
            indices.append(all_timestamps.index(timestamp))
    return indices

def finite_diff(data, time):
    diff = np.zeros_like(data)
    diff[:, 0] = (data[:, 1] - data[:, 0]) / (time[1] - time[0])
    diff[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (time[2:] - time[:-2])
    diff[:, -1] = (data[:, -1] - data[:, -2]) / (time[-1] - time[-2])
    return diff


def compute_gait_parameters(timestamps, force_filtered, mks, mks_name):
    """Compute the gait parameters from the cycle data."""
    global FORCE_MIN_THRESHOLD, MASS

    gait_parameters = [None, None, None, None, None]

    force_filtered_R = force_filtered[0, 2, :]
    force_filtered_L = force_filtered[1, 2, :]

    if force_filtered_R.shape[0] != mks.shape[2]:
        print(force_filtered_R.shape, mks.shape)
        raise RuntimeError("I expected the data to be the same shape")

    cycle_start = timestamps[0]
    cycle_end = timestamps[-1]
    cycle_duration = cycle_end - cycle_start

    # The cycle starts when the right foot is on the ground
    toe_off_R_idx = np.where(force_filtered_R > MASS * FORCE_MIN_THRESHOLD * 9.81)[0]
    if len(toe_off_R_idx) > 0:
        toe_off_R_idx = toe_off_R_idx[-1]
    else:
        return gait_parameters  # Skipping

    toe_off_R = timestamps[toe_off_R_idx]
    stance_duration_R = toe_off_R - cycle_start

    # The left stance is splitted in two parts
    nb_half_frames_cycle = int(len(force_filtered_L) * 1/3)
    toe_off_L_idx = np.where(force_filtered_L[:nb_half_frames_cycle] > MASS * FORCE_MIN_THRESHOLD * 9.81)[0]
    if len(toe_off_L_idx) > 0:
        toe_off_L_idx = toe_off_L_idx[-1]
    else:
        return gait_parameters  # Skipping

    toe_off_L = timestamps[toe_off_L_idx]
    heel_strike_L_idx = np.where(force_filtered_L[nb_half_frames_cycle:] > MASS * FORCE_MIN_THRESHOLD * 9.81)[0]
    if len(heel_strike_L_idx) > 0:
        heel_strike_L_idx = heel_strike_L_idx[0] + nb_half_frames_cycle
    else:
        return gait_parameters  # Skipping

    heel_strike_L = timestamps[heel_strike_L_idx]
    stance_duration_L = (toe_off_L - cycle_start) + (cycle_end - heel_strike_L)

    angle_marker_index_R = [mks_name.index("RSPH"), mks_name.index("RLM")]
    ankle_position_start_R = mks[angle_marker_index_R, :, 0]
    ankle_position_stop_R = mks[angle_marker_index_R, :, toe_off_R_idx]
    step_distance_R = np.linalg.norm(ankle_position_stop_R - ankle_position_start_R)

    angle_marker_index_L = [mks_name.index("LSPH"), mks_name.index("LLM")]
    ankle_position_start_L = mks[angle_marker_index_L, :, heel_strike_L_idx]
    ankle_position_stop_L = mks[angle_marker_index_L, :, toe_off_R_idx]
    step_distance_L = np.linalg.norm(ankle_position_stop_L - ankle_position_start_L)

    gait_parameters = [cycle_duration, stance_duration_R, stance_duration_L, step_distance_R, step_distance_L]
    return gait_parameters


def nan_filtfilt(b, a, data):
    nan_mask = np.isnan(data)
    if np.all(nan_mask):
        return np.zeros_like(data)

    filtered = np.copy(data)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) > 1:
        filtered[valid_idx] = filtfilt(b, a, data[valid_idx])
    return filtered


def data_filter(self, data, order, sampling_rate, cutoff_freq):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low")

    data = np.asarray(data)
    filtered_data = np.empty_like(data)

    if data.ndim == 2:  # (3, T)
        for i in range(data.shape[0]):
            filtered_data[i, :] = nan_filtfilt(b, a, data[i, :])
    elif data.ndim == 3:  # (3, N, T)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered_data[i, j, :] = nan_filtfilt(b, a, data[i, j, :])
    else:
        raise ValueError("Data must be 2D or 3D.")

    return filtered_data


class DataReceiver:
    """Reçoit les données du serveur TCP et les stocke dans Redis"""

    def __init__(self, server_ip, server_port, read_frequency=MARKER_FREQUENCY):
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
        global IS_REDIS_CONNECTED, redis_client

        PRINT_FREQUENCY = True  # For debugging purposes

        NUMBER_OF_FORCE_DATA = 0
        TIC_FORCE_DATA = 0
        TIC_MARKER_DATA = 0
        last_marker_frame = np.empty((16, 3))
        last_force_frame = np.empty((2, 9, 0))
        redis_client.flushdb()

        self.running = True
        try:
            self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=self.read_frequency)

            while self.running:
                try:
                    if IS_REDIS_CONNECTED:
                        received_data = self.tcp_client.get_data_from_server(
                            command=["timestamp", "force", "mks", "mks_name"]
                        )

                        # Reformat data because pickle (in the tcp server) does not support numpy arrays
                        markers_frame = np.array([m for m in received_data["mks"]])

                        there_are_no_forces = (
                            len(received_data["force"][0]) == 0 and len(received_data["force"][1]) == 0
                        )
                        if there_are_no_forces:
                            if last_force_frame.shape[2] < 38 and last_force_frame.shape[2] > 42:
                                raise RuntimeError(
                                    "This code was hacked knowing that there are always 39 or 40 forces per frame and that on frame out of two do not have any forces."
                                )

                            mean_forces_this_frame = np.nanmean(last_force_frame[:, :, 20:], axis=2)
                            forces = np.ones((2, 9, 40))
                            forces[:, :, :] = np.nan

                        else:
                            force_0 = np.array([f for f in received_data["force"][0]])
                            force_1 = np.array([f for f in received_data["force"][1]])
                            forces = np.array([force_0, force_1])
                            if PRINT_FREQUENCY:
                                NUMBER_OF_FORCE_DATA += received_data["force"][0].shape[1]
                                if NUMBER_OF_FORCE_DATA % 1000 == 0:
                                    TOC_FORCE_DATA = datetime.datetime.timestamp(datetime.datetime.now())
                                    elapsed_time = TOC_FORCE_DATA - TIC_FORCE_DATA
                                    print(elapsed_time, "  ----  ", NUMBER_OF_FORCE_DATA / elapsed_time, " Hz")
                                    TIC_FORCE_DATA = TOC_FORCE_DATA
                                    NUMBER_OF_FORCE_DATA = 0

                            mean_forces_this_frame = np.nanmean(forces[:, :, :20], axis=2)

                        if float(np.nansum(markers_frame)) == 0.0:
                            # print("skipping - All markers are NaNs")
                            continue
                        elif np.all(markers_frame == last_marker_frame):
                            # print("skipping - Not a new frame")
                            continue

                        """ Data markers """
                        if self.mks_name is None:
                            self.mks_name = received_data["mks_name"]
                            safe_redis_operation(redis_client.rpush, "mks_name", json.dumps(self.mks_name))
                            safe_redis_operation(redis_client.ltrim, "mks_name", -FRAME_BUFFER_LENGTH, -1)

                        # Créer un identifiant unique (timestamp + compteur)
                        self.frame_counter += 1
                        if PRINT_FREQUENCY:
                            if self.frame_counter % 1000 == 0:
                                TOC_MARKER_DATA = datetime.datetime.timestamp(datetime.datetime.now())
                                elapsed_time = TOC_MARKER_DATA - TIC_MARKER_DATA
                                print(
                                    f"Frame Counter: {self.frame_counter}",
                                    "  ----  ",
                                    100 / elapsed_time,
                                    " Hz",
                                )
                                TIC_MARKER_DATA = TOC_MARKER_DATA

                        # Stocker le timestamp de la mesure puisque la frequence d'acquisition fluctue
                        safe_redis_operation(redis_client.rpush, "timestamp", received_data["timestamp"])
                        safe_redis_operation(redis_client.ltrim, "timestamp", -FRAME_BUFFER_LENGTH, -1)

                        safe_redis_operation(redis_client.rpush, "mks", json.dumps(markers_frame.tolist()))
                        safe_redis_operation(redis_client.ltrim, "mks", -FRAME_BUFFER_LENGTH, -1)

                        safe_redis_operation(redis_client.rpush, "force", json.dumps(mean_forces_this_frame.tolist()))
                        safe_redis_operation(redis_client.ltrim, "force", -FRAME_BUFFER_LENGTH, -1)

                        self.data_received = "Data received successfully"

                        # Flush the forces_this_frame buffer for the next frame
                        last_marker_frame = markers_frame
                        last_force_frame = forces

                except Exception as e:
                    logging.error(f"Erreur dans DataReceiver: {e}")
                    redis_client.flushdb()
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
        self.processed_frame_timestamps = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        self.processed_cycles = deque(maxlen=2 * CYCLE_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"
        self.cycle_counter = 0  # For the detection of cycles
        self.cycle_idx = 0  # For the treatment of the cycles
        self.cycle_start_id = None

    def start_processing(self):
        global PROCESS_ID_IK, IS_REDIS_CONNECTED
        self.running = True

        while self.running:
            try:
                if IS_REDIS_CONNECTED and PROCESS_ID_IK:
                    self.process()
                    self.processing_complete = "Processing complete"

                # Without the sleep, the Interface is way less responsive (but the whole computer is not slowed)
                time.sleep(0.1)  # Réduire la fréquence de traitement
            except Exception as e:
                logging.error(f"Erreur dans DataProcessor: {e}")
                time.sleep(1)

    def identify_cycle_start(self, force_filtered):
        # print("Identifying cycle start...")
        current_cycle_idx = np.ones((force_filtered.shape[1],)) * self.cycle_counter

        right_foot_on_ground_idx = force_filtered[2, :] > FORCE_MIN_THRESHOLD * MASS * 9.81
        right_foot_on_ground_idx = np.astype(right_foot_on_ground_idx, int)
        heel_strike_idx = np.where(np.diff(right_foot_on_ground_idx) == 1)[0] + 1
        toe_off_idx = np.where(np.diff(right_foot_on_ground_idx) == -1)[0] + 1

        if 1 in heel_strike_idx:
            heel_strike_idx = heel_strike_idx[heel_strike_idx != 1]

        # Identification : OK
        # plt.figure()
        # plt.plot(force_filtered[2, :])
        # for frame in heel_strike_idx:
        #     plt.axvline(x=frame, color='red', linestyle='--', label='Heel Strike')
        # for frame in toe_off_idx:
        #     plt.axvline(x=frame, color='green', linestyle='--', label='Toe Off')
        # plt.savefig("cycle_identification.png")
        # plt.show()

        for i_cycle in range(heel_strike_idx.shape[0]):
            self.cycle_counter += 1
            if heel_strike_idx.shape[0] > i_cycle + 1:
                current_cycle_idx[heel_strike_idx[i_cycle] : heel_strike_idx[i_cycle + 1]] = self.cycle_counter
            else:
                current_cycle_idx[heel_strike_idx[i_cycle] :] = self.cycle_counter

        return heel_strike_idx

    def process(self):
        try:
            global MODEL

            timestamps_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
            new_indices, new_frame_timestamps = get_new_indices(timestamps_all, self.processed_frame_timestamps, print_option=False)

            if new_frame_timestamps.shape[0] > 50:
                forces_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)
                mks_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("mks", 0, -1)]
                mks_all = np.array(mks_all).transpose(1, 2, 0)
                mks_name = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("mks_name", 0, -1)][0]

                if mks_all.shape[2] != len(timestamps_all) or forces_all.shape[2] != len(timestamps_all):
                    # logging.info("Les données de mks et forces ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    return

                if mks_all.shape[0] != 16 or len(mks_name) != 16 or MODEL.nbMarkers() != 16:
                    raise RuntimeError("The model used or the labeled markers are not from the 16 markerset.")

                forces = forces_all[:, :, new_indices]
                force_filtered_R = data_filter(forces[0, 0:3, :], 2, MARKER_FREQUENCY, 10)

                heel_strike_idx = self.identify_cycle_start(force_filtered_R)

                if heel_strike_idx.shape[0] > 0:
                    if self.cycle_start_id is None:
                        # We skip on purpose everything before the first heel strike is detected
                        self.cycle_start_id = str(new_frame_timestamps[heel_strike_idx[0]])
                        # print("initialization : start id = ", self.cycle_start_id)
                        self.processed_frame_timestamps.extend(new_frame_timestamps[: heel_strike_idx[0]])
                    else:
                        cycle_stop_id = str(new_frame_timestamps[heel_strike_idx[0]])

                        # Récupérer les données pour ce cycle uniquement
                        if self.cycle_start_id not in timestamps_all:
                            logging.info(
                                f"Cycle start ID {self.cycle_start_id} not found in all frame IDs. "
                                f"Skipping this cycle."
                            )
                            self.cycle_start_id = None
                            return
                        cycle_start_idx = timestamps_all.index(self.cycle_start_id)
                        cycle_stop_idx = timestamps_all.index(cycle_stop_id)

                        idx = 0
                        while cycle_stop_idx - cycle_start_idx < 30:
                            idx += 1
                            if len(heel_strike_idx) > idx + 1:
                                cycle_stop_id = str(new_frame_timestamps[heel_strike_idx[idx]])
                                cycle_stop_idx = timestamps_all.index(cycle_stop_id)
                                # print("start id: ", self.cycle_start_id, " / stop id: ", cycle_stop_id)
                            else:
                                # logging.info("Cycle trop court, pas de traitement.")
                                self.cycle_start_id = None
                                return

                        # print("start id: ", self.cycle_start_id, " / stop id: ", cycle_stop_id)
                        # print("start idx: ", cycle_start_idx, " / stop idx: ", cycle_stop_idx)
                        print("cycle idx : ", self.cycle_idx)

                        mks = mks_all[:, :, cycle_start_idx : cycle_stop_idx + 1]
                        forces = forces_all[:, :, cycle_start_idx : cycle_stop_idx + 1]
                        timestamps = timestamps_all[cycle_start_idx : cycle_stop_idx + 1]

                        if MODEL is not None:
                            # print("Calcul IK/ID...")

                            q, qdot, qddot = self.inverse_kinematics(MODEL, mks, mks_name, timestamps)
                            if q is not None:
                                tau, force_filtered = self.inverse_dynamics(MODEL, forces, q, qdot, qddot)

                                gait_parameters = compute_gait_parameters(
                                    timestamps, force_filtered, mks, mks_name
                                )

                                # print("q envoyé: ", q.shape)
                                q = q.tolist()
                                qdot = qdot.tolist()
                                qddot = qddot.tolist()
                                if tau is not None:
                                    tau = tau.tolist()

                            else:
                                tau = None
                                gait_parameters = [None, None, None, None, None]

                            # Stocker les résultats dans Redis
                            # Stocker l'indice dans une liste séparée pour suivre l'ordre
                            safe_redis_operation(redis_client.rpush, "cycle_idx", self.cycle_idx)
                            safe_redis_operation(redis_client.ltrim, "cycle_idx", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "q", json.dumps(q))
                            safe_redis_operation(redis_client.ltrim, "q", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "qdot", json.dumps(qdot))
                            safe_redis_operation(redis_client.ltrim, "qdot", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "qddot", json.dumps(qddot))
                            safe_redis_operation(redis_client.ltrim, "qddot", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "tau", json.dumps(tau))
                            safe_redis_operation(redis_client.ltrim, "tau", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "gait_parameters", json.dumps(gait_parameters))
                            safe_redis_operation(redis_client.ltrim, "gait_parameters", -CYCLE_BUFFER_LENGTH, -1)

                            # Also add again the original data split by cycle
                            safe_redis_operation(redis_client.rpush, "mks_cycle", json.dumps(mks.tolist()))
                            safe_redis_operation(redis_client.ltrim, "mks_cycle", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "forces_cycle", json.dumps(forces.tolist()))
                            safe_redis_operation(redis_client.ltrim, "forces_cycle", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "timestamps_cycle", json.dumps(timestamps.tolist()))
                            safe_redis_operation(redis_client.ltrim, "timestamps_cycle", -CYCLE_BUFFER_LENGTH, -1)

                        self.cycle_start_id = cycle_stop_id
                        self.processed_frame_timestamps.extend(timestamps_all[cycle_start_idx : cycle_stop_idx + 1])
                        self.cycle_idx += 1

        except Exception as e:
            logging.error(f"Erreur lors du traitement des données: {e}")

    def inverse_kinematics(self, model: biorbd.Model, mks, labels, time):
        try:
            marker_names = tuple(n.to_string() for n in MODEL.technicalMarkerNames())
            index_in_c3d = np.array(tuple(labels.index(name) if name in labels else -1 for name in marker_names))
            mks_to_filter = mks[index_in_c3d[index_in_c3d >= 0], :3, :].transpose(1, 0, 2)

            # Apply the filter to each coordinate (x, y, z) over time
            smoothed_mks = data_filter(data=mks_to_filter, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)

            # Store the result
            ik = biorbd.InverseKinematics(model, smoothed_mks)
            ik.solve(method="trf")
            q = ik.q
            q = data_filter(q, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)
            qdot = finite_diff(q, time)
            qddot = finite_diff(qdot, time)
            return q, qdot, qddot
        except Exception as e:
            logging.error(f"Erreur dans inverse_kinematics: {e}")
            return None, None, None

    def inverse_dynamics(self, model: biorbd.Model, force, q, qdot, qddot):
        try:
            num_contacts = len(force)
            num_frames = force[0].shape[1]
            platform_origin = [[0.78485, 0.7825, 0.0], [0.78485, 0.2385, 0.0]]
            force_filtered = np.zeros((num_contacts, 3, num_frames))
            moment_filtered = np.zeros((num_contacts, 3, num_frames))
            tau_data = np.zeros((model.nbQ(), num_frames))

            for contact_idx in range(num_contacts):
                force_filtered[contact_idx] = data_filter(force[contact_idx][0:3], 2, MARKER_FREQUENCY, 10)
                moment_filtered[contact_idx] = data_filter(force[contact_idx][3:6], 4, MARKER_FREQUENCY, 10)

            for i in range(num_frames):
                ext_load = model.externalForceSet()
                for contact_idx in range(num_contacts):
                    fz = force_filtered[contact_idx, 2, i]
                    if fz > 30:
                        force_vec = force_filtered[contact_idx, :, i]
                        moment_vec = moment_filtered[contact_idx, :, i] / 1000
                        spatial_vector = np.concatenate((moment_vec, force_vec))
                        point_app = platform_origin[contact_idx]
                        segment_name = "LFoot" if contact_idx == 0 else "RFoot"
                        ext_load.add(biorbd.String(segment_name), spatial_vector, np.array(point_app))

                tau = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], ext_load)
                tau_data[:, i] = tau.to_array()

            return tau_data, force_filtered
        except Exception as e:
            logging.error(f"Erreur dans inverse_dynamics: {e}")
            return None

    def stop(self):
        self.running = False
        self.wait()

class QProcessor:
    """Traite les données pour calculer les angles et moments articulaires"""

    def __init__(self):
        super().__init__()
        self.running = True
        self.processed_frame_timestamps = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"

    def start_processing(self):
        global PROCESS_ID_IK, IS_REDIS_CONNECTED
        self.running = True

        while self.running:
            try:
                if IS_REDIS_CONNECTED and PROCESS_ID_IK:
                    self.process()
                    self.processing_complete = "Processing complete"

                # Without the sleep, the Interface is way less responsive (but the whole computer is not slowed)
                time.sleep(0.1)  # Réduire la fréquence de traitement
            except Exception as e:
                logging.error(f"Erreur dans QProcessor: {e}")
                time.sleep(1)

    def process(self):
        try:
            global MODEL

            timestamps_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
            new_indices, new_frame_timestamps = get_new_indices(timestamps_all,
                                                                        self.processed_frame_timestamps,
                                                                        print_option=False)

            if new_frame_timestamps.shape[0] > 0:
                mks_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("mks", 0, -1)]
                mks_all = np.array(mks_all).transpose(1, 2, 0)
                mks_name = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("mks_name", 0, -1)][0]

                if mks_all.shape[2] != len(timestamps_all):
                    # logging.info("Les données de mks et forces ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    return

                if mks_all.shape[0] != 16 or len(mks_name) != 16 or MODEL.nbMarkers() != 16:
                    raise RuntimeError("The model used or the labeled markers are not from the 16 markerset.")

                self.processed_frame_timestamps.extend(new_frame_timestamps)

                mks = mks_all[:, :, new_indices]
                timestamps_q = timestamps_all[new_indices]

                if MODEL is not None:
                    # print("Calcul IK/ID...")

                    q = self.inverse_kinematics(MODEL, mks, mks_name)
                    if q is not None:
                        q = q.tolist()

                    safe_redis_operation(redis_client.rpush, "q", json.dumps(q))
                    safe_redis_operation(redis_client.ltrim, "q", -CYCLE_BUFFER_LENGTH, -1)

                    safe_redis_operation(redis_client.rpush, "timestamp_q", json.dumps(timestamps_q))
                    safe_redis_operation(redis_client.ltrim, "timestamp_q", -CYCLE_BUFFER_LENGTH, -1)

                self.processed_frame_timestamps.extend(new_frame_timestamps)

        except Exception as e:
            logging.error(f"Erreur lors du traitement des données: {e}")

    def inverse_kinematics(self, model: biorbd.Model, mks, labels):
        try:
            marker_names = tuple(n.to_string() for n in MODEL.technicalMarkerNames())
            index_in_c3d = np.array(tuple(labels.index(name) if name in labels else -1 for name in marker_names))
            mks_to_filter = mks[index_in_c3d[index_in_c3d >= 0], :3, :].transpose(1, 0, 2)

            # Apply the filter to each coordinate (x, y, z) over time
            smoothed_mks = data_filter(data=mks_to_filter, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)

            # Store the result
            ik = biorbd.InverseKinematics(model, smoothed_mks)
            ik.solve(method="trf")
            q = ik.q
            q = data_filter(q, cutoff_freq=10, sampling_rate=MARKER_FREQUENCY, order=4)
            return q
        except Exception as e:
            logging.error(f"Erreur dans inverse_kinematics Q: {e}")
            return None

    def stop(self):
        self.running = False
        self.wait()


class TauProcessor:
    """Traite les données pour calculer les angles et moments articulaires"""

    def __init__(self):
        super().__init__()
        self.running = True
        self.processed_frame_timestamps = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"

    def start_processing(self):
        global PROCESS_ID_IK, IS_REDIS_CONNECTED
        self.running = True

        while self.running:
            try:
                if IS_REDIS_CONNECTED and PROCESS_ID_IK:
                    self.process()
                    self.processing_complete = "Processing complete"

                # Without the sleep, the Interface is way less responsive (but the whole computer is not slowed)
                time.sleep(0.1)  # Réduire la fréquence de traitement
            except Exception as e:
                logging.error(f"Erreur dans TauProcessor: {e}")
                time.sleep(1)

    def process(self):
        try:
            global MODEL

            timestamps_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
            new_indices, new_frame_timestamps = get_new_indices(timestamps_all, self.processed_frame_timestamps, print_option=False)

            if new_frame_timestamps.shape[0] > 0:
                forces_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)
                q_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("q", 0, -1)]
                q_all = np.array(q_all).transpose(1, 2, 0)
                timestamps = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
                timestamps_q_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp_q", 0, -1)]
                timestamps_q_all = np.array(timestamps_q_all)

                if q_all.shape[1] != timestamps_q_all.shape[0]:
                    # logging.info("Les données de q ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    return

                force_indices = get_indices_of_these_timestamps(new_frame_timestamps, timestamps)

                forces = forces_all[:, :, force_indices]
                q = q_all[:, new_indices]
                timestamps_q = timestamps_q_all[new_indices]
                self.processed_frame_timestamps.extend(new_frame_timestamps)

                if MODEL is not None:
                    # print("Calcul IK/ID...")
                    if q is not None and q_all.shape[1] > 2:
                        start_idx = new_indices[0]
                        end_idx = new_indices[-1]

                        # Compute qdot (one frame late)
                        qdot = (q_all[:, start_idx: end_idx] - q_all[:, start_idx-2: end_idx-2]) / (timestamps_q_all[start_idx: end_idx] - timestamps_q[start_idx-2: end_idx-2])
                        timestamps_qdot = timestamps_q_all[start_idx-1: end_idx-1]

                        # Put qdot in database
                        safe_redis_operation(redis_client.rpush, "qdot", json.dumps(qdot.tolist()))
                        safe_redis_operation(redis_client.ltrim, "qdot", -CYCLE_BUFFER_LENGTH, -1)

                        safe_redis_operation(redis_client.rpush, "timestamps_qdot", json.dumps(timestamps_qdot))
                        safe_redis_operation(redis_client.ltrim, "timestamps_qdot", -CYCLE_BUFFER_LENGTH, -1)

                        # Pull qdot from database to get the previous ones
                        qdot_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("qdot", 0, -1)]
                        qdot_all = np.array(qdot_all).transpose(1, 2, 0)
                        all_qdot_timestamps = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamps_qdot", 0, -1)]
                        if qdot_all.shape[1] != len(all_qdot_timestamps):
                            return
                        qdot_indices = get_indices_of_these_timestamps(timestamps_qdot, all_qdot_timestamps)

                        start_idx = qdot_indices[0]
                        end_idx = qdot_indices[-1]
                        if qdot_all.shape[1] > 2:
                            # Compute qddot (two frames late)
                            qddot = (qdot_all[:, start_idx: end_idx] - qdot_all[:, start_idx-2: end_idx-2]) / (all_qdot_timestamps[start_idx: end_idx] - all_qdot_timestamps[start_idx-2: end_idx-2])
                            timestamps_qddot = all_qdot_timestamps[start_idx-1: end_idx-1]

                            # Put qddot in database
                            safe_redis_operation(redis_client.rpush, "qddot", json.dumps(qddot.tolist()))
                            safe_redis_operation(redis_client.ltrim, "qddot", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "timestamps_qddot", json.dumps(timestamps_qddot))
                            safe_redis_operation(redis_client.ltrim, "timestamps_qddot", -CYCLE_BUFFER_LENGTH, -1)

                            # Compute Tau from the data computed at the same timestamp
                            q_indices = get_indices_of_these_timestamps(timestamps_qddot, timestamps_q_all)
                            qdot_indices = get_indices_of_these_timestamps(timestamps_qddot, all_qdot_timestamps)
                            force_indices = get_indices_of_these_timestamps(timestamps_qddot, timestamps)

                            tau, force_filtered = self.inverse_dynamics(MODEL, forces_all[:, :, force_indices], q_all[:, q_indices], qdot[:, qdot_indices], qddot)

                            # Put tau in database
                            safe_redis_operation(redis_client.rpush, "tau", json.dumps(tau.tolist()))
                            safe_redis_operation(redis_client.ltrim, "tau", -CYCLE_BUFFER_LENGTH, -1)

                            safe_redis_operation(redis_client.rpush, "timestamps_tau", json.dumps(timestamps_qddot))
                            safe_redis_operation(redis_client.ltrim, "timestamps_tau", -CYCLE_BUFFER_LENGTH, -1)

        except Exception as e:
            logging.error(f"Erreur dans TauProcessor : {e}")

    def inverse_dynamics(self, model: biorbd.Model, force, q, qdot, qddot):
        try:
            # TODO: verify this step !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            num_contacts = len(force)
            num_frames = force[0].shape[1]
            platform_origin = [[0.78485, 0.7825, 0.0], [0.78485, 0.2385, 0.0]]  # Not good
            force_filtered = np.zeros((num_contacts, 3, num_frames))
            moment_filtered = np.zeros((num_contacts, 3, num_frames))
            tau_data = np.zeros((model.nbQ(), num_frames))

            for contact_idx in range(num_contacts):
                force_filtered[contact_idx] = data_filter(force[contact_idx][0:3], 2, MARKER_FREQUENCY, 10)
                moment_filtered[contact_idx] = data_filter(force[contact_idx][3:6], 4, MARKER_FREQUENCY, 10)

            for i in range(num_frames):
                ext_load = model.externalForceSet()
                for contact_idx in range(num_contacts):
                    fz = force_filtered[contact_idx, 2, i]
                    if fz > 30:
                        force_vec = force_filtered[contact_idx, :, i]
                        moment_vec = moment_filtered[contact_idx, :, i] / 1000
                        spatial_vector = np.concatenate((moment_vec, force_vec))
                        point_app = platform_origin[contact_idx]
                        segment_name = "LFoot" if contact_idx == 0 else "RFoot"
                        ext_load.add(biorbd.String(segment_name), spatial_vector, np.array(point_app))

                tau = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], ext_load)
                tau_data[:, i] = tau.to_array()

            return tau_data, force_filtered
        except Exception as e:
            logging.error(f"Erreur dans inverse_dynamics: {e}")
            return None

    def stop(self):
        self.running = False
        self.wait()

class BayesianOptimizer:
    """Traite les données pour determiner quels parametres de stimulation essayer"""

    def __init__(self):
        super().__init__()
        self.running = True

        # self.processed_frame_timestamps = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        # self.processed_cycles = deque(maxlen=2 * CYCLE_BUFFER_LENGTH)
        self.processing_complete = "Not initialized"
        self.current_iteration = None
        self.current_cycle = None

        # Optimization parameters
        # Define the variable bounds
        self.bounds = [
            Real(20, 50, name="R_frequency"),  # Hz
            Real(10, 20, name="R_intensity"),  # mA
            Real(200, 500, name="R_width"),  # micros
            Real(20, 50, name="L_frequency"),  # Hz
            Real(10, 20, name="L_intensity"),  # mA
            Real(200, 500, name="L_width"),  # micros
        ]

        # Define the objective weightings
        # TODO: Charbie -> how do we chose which objectives to minimize ?
        self.weight_comddot = 1
        self.weight_angular_momentum = 1
        self.weight_enegy = 1
        self.weight_ankle_power = -1

    def start_optimizing(self):
        global IS_REDIS_CONNECTED, RUN_OPTIMISATION

        self.running = True

        while self.running:
            try:
                if IS_REDIS_CONNECTED and RUN_OPTIMISATION:
                    """Perform Bayesian optimization using Gaussian Processes."""

                    # gp_minimize will try to find the minimal value of the objective function.
                    result = gp_minimize(
                        func=lambda stimulation_params: self.make_an_iteration(stimulation_params),
                        dimensions=self.bounds,
                        n_calls=100,  # number of evaluations of f
                        acq_func="LCB",  # "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
                        kappa=5,  # *
                        random_state=0,  # *
                        n_jobs=1,
                    )  # x0, y0, kappa[exploitation, exploration], xi [minimal improvement default 0.01]

                    # TODO: allow for different chanel (now right = 1 and left = 5)

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

    def set_stimulation_parameters(self, stimulation_params):
        global IS_REDIS_CONNECTED

        # Current values of the optimized FES parameters
        R_frequency = stimulation_params[0]
        R_intensity = stimulation_params[1]
        R_width = stimulation_params[2]
        L_frequency = stimulation_params[3]
        L_intensity = stimulation_params[4]
        L_width = stimulation_params[5]

        stimulator_parameters = {}
        stimulator_parameters["1"] = {
            "name": f"Canal 1",
            "amplitude": R_intensity,
            "pulse_width": R_width,
            "frequency": R_frequency,
            "mode": "SINGLE",
        }
        stimulator_parameters["5"] = {
            "name": f"Canal 5",
            "amplitude": L_intensity,
            "pulse_width": L_width,
            "frequency": L_frequency,
            "mode": "SINGLE",
        }

        if IS_REDIS_CONNECTED:
            try:
                safe_redis_operation(redis_client.rpush, "stimulation_parameters", json.dumps(stimulator_parameters))
                safe_redis_operation(redis_client.ltrim, "stimulation_parameters", -FRAME_BUFFER_LENGTH, -1)
                logging.info(f"Paramètres de stimulation mis à jour par l'optimisation Bayesienne: {stimulation_params}")
            except Exception as e:
                logging.error(f"Erreur lors de la mise à jour des paramètres: {e}")

    def get_cycle_data(self):

        no_new_data = True
        while no_new_data:
            cycle_indices = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("cycle_idx", 0, -1)]
            q_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("q", 0, -1)]
            qdot_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("qdot", 0, -1)]
            qddot_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("qddot", 0, -1)]
            tau_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("tau", 0, -1)]
            gait_parameters_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("gait_parameters", 0, -1)]

            if (
                len(q_all) != len(cycle_indices)
                or len(qdot_all) != len(cycle_indices)
                or len(qddot_all) != len(cycle_indices)
                or len(tau_all) != len(cycle_indices)
                or len(gait_parameters_all) != len(cycle_indices)
            ):
                # We are in a weird state, it is better to wait for the next loop
                continue

            if len(q_all) > 0 and len(cycle_indices) > 0:
                if self.current_cycle is None:
                    self.current_cycle = cycle_indices[0]
                elif self.current_cycle < cycle_indices[-1]:
                    self.current_cycle += 1
                else:
                    continue

                this_cycle_index = cycle_indices.index(self.current_cycle)
                q = np.array(q_all[this_cycle_index])
                qdot = np.array(qdot_all[this_cycle_index])
                qddot = np.array(qddot_all[this_cycle_index])
                tau = np.array(tau_all[this_cycle_index])
                gait_parameters = gait_parameters_all[this_cycle_index]
                no_new_data = False

        return q, qdot, qddot, tau, gait_parameters

    def make_an_iteration(self, stimulation_params):
        global START_STIMULATION, STOP_STIMULATOR

        if self.current_iteration is None:
            self.current_iteration = 0
        else:
            self.current_iteration += 1

        # Set the parameter values to test this iteration
        self.set_stimulation_parameters(stimulation_params)

        # Stimulate
        START_STIMULATION = True

        # Collect data while waiting for the subject to get a stable walking pattern with these parameters
        cycles = {
            "q": [],
            "qdot": [],
            "qddot": [],
            "tau": [],
            "cycle_duration": [],
            "stance_duration_R": [],
            "stance_duration_L": [],
            "step_distance_R": [],
            "step_distance_L": [],
            "nb_frames": [],
        }

        stable = False
        while not stable:
            q_new, qdot_new, qddot_new, tau_new, gait_parameters_new = self.get_cycle_data()

            cycles["q"] += [q_new]
            cycles["qdot"] += [qdot_new]
            cycles["qddot"] += [qddot_new]
            cycles["tau"] += [tau_new]
            cycles["cycle_duration"] += gait_parameters_new[0]
            cycles["stance_duration_R"] += gait_parameters_new[1]
            cycles["stance_duration_L"] += gait_parameters_new[2]
            cycles["step_distance_R"] += gait_parameters_new[3]
            cycles["step_distance_L"] += gait_parameters_new[4]
            cycles["nb_frames"] += q_new.shape[1]
            if len(cycles["q"]) > 10:
                # Compute the std of the last 10 cycles
                cycle_duration_std = np.nanstd(cycles["cycle_duration"][-10:])
                stance_duration_R_std = np.nanstd(cycles["stance_duration_R"][-10:])
                stance_duration_L_std = np.nanstd(cycles["stance_duration_L"][-10:])
                step_distance_R_std = np.nanstd(cycles["step_distance_R"][-10:])
                step_distance_L_std = np.nanstd(cycles["step_distance_L"][-10:])

                # Check if the last 10 cycles are stable
                # TODO !!!
                stable = True
                # stable = (
                # cycle_duration_std < 0.05 * np.nanmean(cycles["cycle_duration"][-10:])
                # and stance_duration_R_std < 0.05 * np.nanmean(cycles["stance_duration_R"][-10:])
                # and stance_duration_L_std < 0.05 * np.nanmean(cycles["stance_duration_L"][-10:])
                # and step_distance_R_std < 0.05 * np.nanmean(cycles["step_distance_R"][-10:])
                # and step_distance_L_std < 0.05 * np.nanmean(cycles["step_distance_L"][-10:])
                # )

        # Stop the stimulation
        STOP_STIMULATOR = True

        # Compute the mean cycle
        q_mean, qdot_mean, qddot_mean, tau_mean = self.compute_mean_cycle(cycles)

        # Compute objective values
        R_intensity = stimulation_params[1]
        L_intensity = stimulation_params[4]
        total_cost, detailed_cost = self.objective(q_mean, qdot_mean, qddot_mean, tau_mean, R_intensity, L_intensity)

        self.save_iteration_data(cycles, q_mean, qdot_mean, qddot_mean, tau_mean, stimulation_params, detailed_cost)

        return total_cost

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
        ankle_index = [DOF_CORR["RAnkle"][0], DOF_CORR["LAnkle"][0]]
        sum_ankles = np.sum(np.abs(tau[ankle_index, :] * qdot[ankle_index, :]), axis=0)
        return np.trapezoid(sum_ankles, x=time_vector)

    def objective(self, q, qdot, qddot, tau, R_intensity, L_intensity):
        global MODEL

        nb_frames = q.shape[1]
        read_frequency = MARKER_FREQUENCY  # Hz
        time_vector = np.linspace(0, (nb_frames - 1) * 1 / read_frequency, nb_frames)

        comddot = self.compute_com_acceleration(MODEL, q, qdot, qddot)
        angular_momentum = self.compute_angular_momentum(MODEL, q, qdot, qddot)
        energy_human = self.compute_energy(qdot, tau, R_intensity, L_intensity, time_vector)
        power_ankle = self.compute_ankle_power(qdot, tau, time_vector)

        comddot_cost = self.weight_comddot * comddot
        angular_momentum_cost = self.weight_angular_momentum * angular_momentum
        energy_cost = self.weight_enegy * energy_human
        ankle_power_cost = self.weight_ankle_power * power_ankle

        total_cost = comddot_cost + angular_momentum_cost + energy_cost + ankle_power_cost
        detailed_cost = [comddot_cost, angular_momentum_cost, energy_cost, ankle_power_cost]

        safe_redis_operation(redis_client.rpush, "cost", json.dumps(detailed_cost))
        safe_redis_operation(redis_client.ltrim, "cost", -CYCLE_BUFFER_LENGTH, -1)

        return total_cost, detailed_cost


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
        global SAVE_PATH

        save_file_name = SAVE_PATH + "/optimal_bayesian_parameters.txt"
        with open(save_file_name, "a+") as f:
            f.write(f"\n\n************** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ***************\n\n")
            f.write("Optimal parameters found through Bayesian optimization : \n\n")
            f.write("Frequency right = %.4f\n" % result.x[0])
            f.write("Intensity right = %.4f\n" % result.x[1])
            f.write("Width right = %.4f\n" % result.x[2])
            f.write("Frequency left = %.4f\n" % result.x[3])
            f.write("Intensity left = %.4f\n" % result.x[4])
            f.write("Width left = %.4f\n" % result.x[5])
            f.write("\nOptimal cost function value = %.4f\n" % result.fun)
        return

    def save_iteration_data(self, cycles, q_mean, qdot_mean, qddot_mean, tau_mean, stimulation_params, detailed_cost):
        global SAVE_PATH

        iter_path = SAVE_PATH + "/iterations"
        if not os.path.exists(iter_path):
            os.makedirs(iter_path)

        with open(f"{iter_path}/iteration_{self.current_iteration}.pkl", 'wb') as f:
            data = {
                "cycles": cycles,
                "qdq_meanot": q_mean,
                "qdot_mean": qdot_mean,
                "qddot_mean": qddot_mean,
                "tau_mean": tau_mean,
                "stimulation_params": stimulation_params,
            }
            pickle.dump(data, f)


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
        self.processed_frame_timestamps = deque(maxlen=2 * FRAME_BUFFER_LENGTH)
        self.data_received = "Not initialized"
        self.fyr = None
        self.fzr = None
        self.fyl = None
        self.fzl = None
        self.should_send_stim = False

    def start_processing(self):
        global ACTIVATE_STIMULATOR, START_STIMULATION, STOP_STIMULATOR, IS_REDIS_CONNECTED

        while self.running:
            try:
                if IS_REDIS_CONNECTED:

                    if ACTIVATE_STIMULATOR:
                        self.activate_stimulator()
                        ACTIVATE_STIMULATOR = False

                    if START_STIMULATION:
                        self.call_start_stimulation(self.last_channels)
                        START_STIMULATION = False
                        self.should_send_stim = True

                    if STOP_STIMULATOR:
                        self.stop_stimulator()
                        STOP_STIMULATOR = False
                        self.should_send_stim = False

                    self.stimulation_process()
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Erreur dans StimulationProcessor: {e}")
                time.sleep(0.01)

    def stimulation_process(self):
        try:
            timestamps_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
            new_indices, new_frame_timestamps = get_new_indices(timestamps_all, self.processed_frame_timestamps, print_option=False)

            if len(new_indices) > 0:
                forces_all = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("force", 0, -1)]
                forces_all = np.array(forces_all).transpose(1, 2, 0)

                if forces_all.shape[2] != len(timestamps_all):
                    # logging.info("Les données de forces ne correspondent pas au nombre d'IDs de frame.")
                    # If we are gathering the data, at the same time as it is written, we might have inconsistent shapes.
                    # In this case, it is better to wait for the next frame to move forward with the processing.
                    # print("Les données de forces ne correspondent pas au nombre d'IDs de frame.")
                    return

                force_data = forces_all[:, :, new_indices]
                self.processed_frame_timestamps.extend(new_frame_timestamps)

                if self.fyr is None:
                    self.fyl = force_data[0][1, :]  # Force Y gauche
                    self.fzl = force_data[0][2, :]  # Force Z gauche
                    self.fyr = force_data[1][1, :]  # Force Y droite
                    self.fzr = force_data[1][2, :]  # Force Z droite
                else:
                    self.fyl = np.concatenate((self.fyl, force_data[0][1, :]), axis=0)
                    self.fzl = np.concatenate((self.fzl, force_data[0][2, :]), axis=0)
                    self.fyr = np.concatenate((self.fyr, force_data[1][1, :]), axis=0)
                    self.fzr = np.concatenate((self.fzr, force_data[1][2, :]), axis=0)
                if self.fyr.shape[0] > 20 and MASS is not None:
                    info_feet = {
                        "right": self.detect_phase_force(self.fyr, self.fzr, self.fzl, 1),
                        "left": self.detect_phase_force(self.fyl, self.fzl, self.fzr, 2),
                    }
                    # print(info_feet, "  ---   ", self.should_send_stim)
                    if self.should_send_stim:
                        self.manage_stimulation(info_feet)
                    self.fyr = self.fyl[-19:]
                    self.fzr = self.fzr[-19:]
                    self.fyl = self.fyl[-19:]
                    self.fzl = self.fzl[-19:]

        except Exception as e:
            logging.error(f"Erreur dans stimulation_process: {e}")

    def detect_phase_force(self, data_force_ap, data_force_v, data_force_opp, foot_num):
        global MASS
        subject_gravity_force = MASS * 9.81

        try:
            info = "nothing"
            data_force_opp = data_force_opp[-30:]

            force_ap_last = data_force_ap[-1]

            subject_standing_on_this_foot = np.nanmean(data_force_v[-10:]) > 0.7 * subject_gravity_force
            the_other_foot_still_touches = np.nanmean(data_force_opp[-10:]) > 50

            antero_posterior_force_is_decreasing = np.nanmean(data_force_ap[-10:] - data_force_ap[-11:-1]) < 0
            antero_posterior_force_is_increasing = not antero_posterior_force_is_decreasing

            small_weight_on_this_foot = np.nanmean(data_force_ap[-10:]) < 0.05 * subject_gravity_force
            antero_posterior_force_is_positive = force_ap_last > -0.01 * subject_gravity_force

            currently_sending_stim_on_this_leg = self.sendStim[foot_num]
            not_currently_sending_stim_on_this_leg = not currently_sending_stim_on_this_leg

            if subject_standing_on_this_foot and not the_other_foot_still_touches:
                antero_posterior_force_is_small = force_ap_last < 0.1 * subject_gravity_force
                last_foot_stimulated_is_the_opposite = self.last_foot_stim is not foot_num
                if (
                    antero_posterior_force_is_small
                    and antero_posterior_force_is_decreasing
                    and not_currently_sending_stim_on_this_leg
                    and last_foot_stimulated_is_the_opposite
                ):
                    info = "StartStim"
                    self.sendStim[foot_num] = True
                    self.last_foot_stim = foot_num

            elif (
                small_weight_on_this_foot
                or (antero_posterior_force_is_increasing and antero_posterior_force_is_positive)
            ) and currently_sending_stim_on_this_leg:
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
            active_channels = []
            active_channels[:] = self.last_channels[:]

            if right == "StartStim":
                for i_chanel in range(1, 5):
                    if i_chanel not in active_channels:
                        active_channels.append(i_chanel)
            elif right == "StopStim":
                for i_chanel in range(1, 5):
                    if i_chanel in active_channels:
                        active_channels.remove(i_chanel)

            if left == "StartStim":
                for i_chanel in range(5, 9):
                    if i_chanel not in active_channels:
                        active_channels.append(i_chanel)
            elif left == "StopStim":
                for i_chanel in range(5, 9):
                    if i_chanel in active_channels:
                        active_channels.remove(i_chanel)

            new_channels = sorted(active_channels)

            if new_channels != self.last_channels:
                if len(new_channels) > 0:
                    self.call_start_stimulation(new_channels)
                    self.stimulation_status = f"Stim send to canal(s): {new_channels}"
                else:
                    self.call_pause_stimulation()
                    self.stimulation_status = "Stim stop"
                self.last_channels[:] = new_channels[:]
        except Exception as e:
            logging.error(f"Erreur dans manage_stimulation: {e}")

    def activate_stimulator(self):
        try:
            if not self.stimulator_is_active:
                # self.stimulator = St(port="COM3", show_log="Status")
                self.stimulator = St(port="COM3", show_log=False)
                self.stimulator_is_active = True
                self.stimulation_status = "Stimulateur activé"
        except Exception as e:
            logging.error(f"Erreur lors de l'activation du stimulateur: {e}")
            self.stimulation_status = f"Erreur: {str(e)}"

    def call_start_stimulation(self, channel_to_send):
        try:
            if not self.stimulator_is_active:
                logging.info("Le stimulateur doit être activé avant de commencer la stimulation.")
                return

            print("start stimulation on channels: ", channel_to_send)

            if self.stimulator_is_sending_stim:
                print("Already sending -> call_pause_stimulation")
                self.call_pause_stimulation()

            stim_params = safe_redis_operation(redis_client.lrange, "stimulation_parameters", 0, -1)

            if stim_params:
                stimulator_parameters = json.loads(stim_params[-1])

                channels_instructions = []
                for channel in stimulator_parameters.keys():
                    channels_instructions += [
                        Channel(
                            mode=stimulator_parameters[channel]["mode"],
                            no_channel=int(channel),
                            amplitude=(
                                stimulator_parameters[channel]["amplitude"] if int(channel) in channel_to_send else 0
                            ),
                            pulse_width=stimulator_parameters[channel]["pulse_width"],
                            frequency=stimulator_parameters[channel]["frequency"],
                            device_type=Device.Rehastimp24,
                            name=stimulator_parameters[channel]["name"],
                        )
                    ]

                if len(channels_instructions) > 0:
                    self.stimulator.init_stimulation(list_channels=channels_instructions)
                    self.stimulator.update_stimulation(upd_list_channels=channels_instructions)
                    self.stimulator.start_stimulation(upd_list_channels=channels_instructions)
                    self.stimulator_is_sending_stim = True
                    self.stimulation_status = f"Stimulation démarrée sur les canaux {channel_to_send}"
                    # print(f"Stimulation démarrée sur les canaux {channel_to_send}")

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
        self.setMinimumSize(1500, 1000)
        self.channel_inputs = {}
        self.num_config = 0
        self.do_look_need_send_stim = False
        self.stimulation_mode = StimulationMode.MANUAL
        self.channel_bounds = {f"Canal {i}": DEFAULT_BOUNDS for i in [1, 2, 5, 6]}
        self.discomfort = 0
        self.which_data_to_plot = {
            "forces": {"active": False, "nb_lines": 4},
            "marker": {"active": False, "nb_lines": 1},
            "tau": {"active": False, "nb_lines": 3},
            "q": {"active": False, "nb_lines": 3},
            "gait_params": {"active": False, "nb_lines": 5},
            "stim_params": {"active": False, "nb_lines": 6},
            "cost": {"active": False, "nb_lines": 4},
        }
        self.graph_axes = {}
        self.graph_plots = {}
        self.initial_time = None

        # Initialize UI components
        self.init_ui()

        # Timer pour mettre à jour les graphes toutes les secondes
        self.graph_update_timer = QTimer(self)
        self.graph_update_timer.timeout.connect(self.update_data_and_graphs)
        self.graph_update_timer.start(100)

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
        self.mass_spin.setValue(MASS)
        ok_mass = QPushButton("OK")
        ok_mass.clicked.connect(lambda: self.update_mass(self.mass_spin.value()))
        mass_layout.addWidget(mass_label)
        mass_layout.addWidget(self.mass_spin)
        mass_layout.addWidget(ok_mass)

        # Modèle
        model_layout = QHBoxLayout()
        self.checkbox_pro_idik = QCheckBox("Processus IK/ID")
        self.checkbox_pro_idik.setChecked(False)
        self.checkbox_pro_idik.setEnabled(False)
        self.checkbox_pro_idik.stateChanged.connect(self.need_process_idik)
        self.model_label = QLabel("Aucun fichier sélectionné")
        model_button = QPushButton("Charger un fichier")
        model_button.clicked.connect(self.upload_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(model_button)
        model_layout.addWidget(self.checkbox_pro_idik)

        # Save path
        save_path_layout = QHBoxLayout()
        self.save_path_label = QLabel("Aucun fichier sélectionné")
        save_path_button = QPushButton("Charger un fichier")
        save_path_button.clicked.connect(self.select_save_path)
        save_path_layout.addWidget(self.save_path_label)
        save_path_layout.addWidget(save_path_button)

        main_layout.addLayout(mass_layout)
        main_layout.addLayout(model_layout)
        main_layout.addChildLayout(save_path_layout)
        groupbox.setLayout(main_layout)
        return groupbox

    def need_process_idik(self):
        """Active le traitement IK/ID"""
        global PROCESS_ID_IK
        PROCESS_ID_IK = self.checkbox_pro_idik.isChecked()
        if PROCESS_ID_IK:
            logging.info("ID / IK started.")
        else:
            logging.info("ID / IK stopped.")

    def upload_file(self):
        """Charge un fichier de modèle"""
        global MODEL_FILE_NAME, MODEL

        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier")
        if file_name:
            try:
                # Load the model file
                MODEL_FILE_NAME = file_name
                MODEL = biorbd.Model(file_name)
                logging.info(f"Fichier modèle chargé: {file_name}")
                self.model_label.setText(file_name.split("/")[-1])

                # Allow for ID/IK processing if the model is loaded
                self.checkbox_pro_idik.setEnabled(True)

            except Exception as e:
                logging.error(f"Erreur lors du chargement du fichier: {str(e)}")
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier: {str(e)}")

    def select_save_path(self):
        """Sélectionne le chemin de sauvegarde pour les données"""
        global SAVE_PATH

        folder_name = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier")
        if folder_name:
            try:
                SAVE_PATH = folder_name
                logging.info(f"Dossier d'enregistrement sélectionné: {folder_name}")
                self.save_path_label.setText(folder_name)
            except Exception as e:
                logging.error(f"Erreur lors de la sélection du dossier: {str(e)}")
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le dossier: {str(e)}")

    def update_mass(self, mass_value):
        """Met à jour la masse du participant"""
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
        for i in [1, 2, 5, 6]:
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
                amplitude_input.setValue(15)
                amplitude_input.setSuffix(" mA")
                pulse_width_input = QSpinBox()
                pulse_width_input.setRange(DEFAULT_BOUNDS["Pulse Width"][0], DEFAULT_BOUNDS["Pulse Width"][1])
                pulse_width_input.setValue(200)
                pulse_width_input.setSuffix(" µs")
                frequency_input = QSpinBox()
                frequency_input.setRange(DEFAULT_BOUNDS["Frequency"][0], DEFAULT_BOUNDS["Frequency"][1])
                frequency_input.setValue(50)
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

    def activate_stimulator(self):
        global ACTIVATE_STIMULATOR
        ACTIVATE_STIMULATOR = True

        # Enable stimulation controls
        self.manual_mode_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self.bayesian_mode_button.setEnabled(True)
        self.ilc_mode_button.setEnabled(False)  # TODO: Charbie -> Implement ILC, for now always disabled

        # Change the connexion status text
        self.connection_status.setText("Statut: Connecté")
        self.connection_status.setStyleSheet("color: black;")

        self.modify_channel_bound_enabling(True)

    def update_stimulation(self):
        if self.stimulator is not None:
            self.stimulator.update_stimulation()

    def manual_optim_chosen(self):
        global RUN_OPTIMISATION
        RUN_OPTIMISATION = False

        self.update_button.setEnabled(True)
        self.start_bayesian_optim_button.setEnabled(False)
        self.stop_bayesian_optim_button.setEnabled(False)
        # TODO: Charbie -> add the ICL buttons

        self.modify_channel_bound_enabling(False)

    def modify_channel_bound_enabling(self, value: bool):
        # Channel Bounds Section
        for i in [1, 2, 5, 6]:
            for i_parameter, parameter_name in enumerate(DEFAULT_BOUNDS.keys()):
                self.channel_bounds_inputs[f"Canal {i}"][parameter_name][0].setEnabled(value)
                self.channel_bounds_inputs[f"Canal {i}"][parameter_name][1].setEnabled(value)

    def bayesian_optim_chosen(self):
        global RUN_OPTIMISATION
        RUN_OPTIMISATION = True

        self.update_button.setEnabled(False)
        self.start_bayesian_optim_button.setEnabled(True)
        self.stop_bayesian_optim_button.setEnabled(True)
        # TODO: Charbie -> add the ICL buttons

        self.modify_channel_bound_enabling(True)

    def ilc_optim_chosen(self):
        global RUN_OPTIMISATION
        RUN_OPTIMISATION = False

        self.update_button.setEnabled(False)
        self.start_bayesian_optim_button.setEnabled(False)
        self.stop_bayesian_optim_button.setEnabled(False)
        # TODO: Charbie -> add the ICL buttons

        self.modify_channel_bound_enabling(False)

    def start_stimulation(self):
        global START_STIMULATION
        START_STIMULATION = True

        if self.stimulation_mode == StimulationMode.MANUAL:
            self.update_button.setEnabled(True)
        elif self.stimulation_mode == StimulationMode.BAYESIAN:
            self.start_bayesian_optim_button.setEnabled(True)
            self.stop_bayesian_optim_button.setEnabled(True)

    def stop_stimulator(self):
        global STOP_STIMULATOR
        STOP_STIMULATOR = True

    def set_discomfort(self, value):
        global DISCOMFORT
        DISCOMFORT = value

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

        self.checkpauseStim = QCheckBox("Stop trying send stim")
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
        global DISCOMFORT

        groupbox = QGroupBox("Stimulation Parameter Mode:")
        layout = QGridLayout()

        # Manual Mode
        self.manual_mode_button = QRadioButton("Manual", self)
        self.manual_mode_button.setChecked(True)
        self.manual_mode_button.toggled.connect(self.manual_optim_chosen)
        self.manual_mode_button.setEnabled(False)

        layout.addWidget(self.manual_mode_button, 0, 0, 1, 1)

        # Bayesian Optimization Mode
        self.bayesian_mode_button = QRadioButton("Bayesian Optimization", self)
        self.bayesian_mode_button.toggled.connect(self.bayesian_optim_chosen)
        self.bayesian_mode_button.setEnabled(False)
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
        self.channel_bounds_inputs = {f"Canal {i}": {} for i in [1, 2, 5, 6]}
        for i in [1, 2, 5, 6]:
            channel_label = QLabel(f"Canal {i} :")
            layout.addWidget(channel_label, 3, i - 1)

            for i_parameter, parameter_name in enumerate(DEFAULT_BOUNDS.keys()):
                if parameter_name not in self.channel_bounds_inputs:
                    self.channel_bounds_inputs[parameter_name] = {}

                channel_min_bound = QSpinBox()
                channel_min_bound.setRange(DEFAULT_BOUNDS[parameter_name][0], DEFAULT_BOUNDS[parameter_name][1])
                channel_min_bound.setValue(DEFAULT_BOUNDS[parameter_name][0])
                channel_min_bound.setEnabled(False)
                channel_max_bound = QSpinBox()
                channel_max_bound.setRange(DEFAULT_BOUNDS[parameter_name][0], DEFAULT_BOUNDS[parameter_name][1])
                channel_max_bound.setValue(DEFAULT_BOUNDS[parameter_name][1])
                channel_max_bound.setEnabled(False)

                self.channel_bounds_inputs[f"Canal {i}"][parameter_name] = [channel_min_bound, channel_max_bound]
                layout.addWidget(channel_min_bound, 4 + 2 * i_parameter, i - 1, 1, 1)
                layout.addWidget(channel_max_bound, 5 + 2 * i_parameter, i - 1, 1, 1)

        amplitude_label = QLabel(" mA")
        layout.addWidget(amplitude_label, 4, i, 1, 1)
        width_label = QLabel(" µs")
        layout.addWidget(width_label, 6, i, 1, 1)
        frequency_label = QLabel(" Hz")
        layout.addWidget(frequency_label, 8, i, 1, 1)

        # stable_cycles_label = QLabel(f"There were <b>{self.num_stable_cycles}</b> stable cycles")
        # layout.addWidget(stable_cycles_label, 0, 5)
        # current_cost_label = QLabel(f"The current cost is <b>{self.current_cost}</b>")
        # layout.addWidget(current_cost_label, 1, 5)
        discomfort_label = QLabel(f"Discomfort  :")
        layout.addWidget(discomfort_label, 2, 4)
        discomfort_box = QSpinBox()
        discomfort_box.setRange(0, 10)
        discomfort_box.setValue(DISCOMFORT)
        layout.addWidget(discomfort_box, 2, 5)
        discomfort_button = QPushButton("Set discomfort")
        discomfort_button.clicked.connect(lambda: self.set_discomfort(discomfort_box.value()))
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
        global START_STIMULATION, STOP_STIMULATOR

        self.do_look_need_send_stim = not self.checkpauseStim.isChecked()
        if self.checkpauseStim.isChecked():
            self.stimulation_status.setText("Stimulation : inactive")
            self.stimulation_status.setStyleSheet("color: gray;")
            STOP_STIMULATOR = True
            START_STIMULATION = False
        else:
            self.stimulation_status.setText("Stimulation : active")
            self.stimulation_status.setStyleSheet("color: black;")
            START_STIMULATION = True
            STOP_STIMULATOR = False

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

        if IS_REDIS_CONNECTED:
            try:
                safe_redis_operation(redis_client.rpush, "stimulation_parameters", json.dumps(stimulator_parameters))
                safe_redis_operation(redis_client.ltrim, "stimulation_parameters", -FRAME_BUFFER_LENGTH, -1)
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
            "color: green;"
            if "démarrée" in message.lower() or "active" in message.lower()
            else "color: red;" if "arrêt" in message.lower() or "erreur" in message.lower() else "color: gray;"
        )

    def create_analysis_group(self):
        """Créer un groupbox pour la sélection des analyses."""
        groupbox = QGroupBox("Sélections d'Analyse")
        layout = QHBoxLayout()

        self.checkboxes_graphs = {}
        for key in self.which_data_to_plot.keys():
            checkbox = QCheckBox(key, self)
            checkbox.stateChanged.connect(self.create_graphs)
            layout.addWidget(checkbox)
            self.checkboxes_graphs[key] = checkbox

        groupbox.setLayout(layout)
        return groupbox

    def update_data_and_graphs(self):
        global redis_client, IS_REDIS_CONNECTED

        if not IS_REDIS_CONNECTED:
            return

        # Parcours des clés de self.which_data_to_plot
        for key in self.which_data_to_plot.keys():
            is_checked = self.which_data_to_plot[key]["active"]
            if not is_checked:
                # Skip if we do not need to show this type of data
                continue

            time_vector = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamp", 0, -1)]
            if key in ["forces", "marker"]:
                if key == "forces":
                    data = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("force", 0, -1)]
                    if data == []:
                        continue
                    data = np.array(data).transpose(1, 2, 0)
                    y_data = [data[0][1, :],
                              data[0][2, :],
                              data[1][1, :],
                              data[1][2, :]]

                elif key == "marker":
                    data = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("mks", 0, -1)]
                    if data == []:
                        continue
                    y_data = [np.array(data)[:, 6, 2]]  # knee marker, z-axis

                n_frames = len(y_data[0])
                if len(time_vector) == n_frames:
                    x_data = np.array(time_vector)
                elif len(time_vector) > n_frames:
                    x_data = np.array(time_vector[:n_frames])
                else:
                    x_data = np.array(time_vector)
                    for i_frame in range(n_frames - len(time_vector)):
                        x_data = np.concatenate((x_data, np.array([x_data[-1] + 1 / MARKER_FREQUENCY])))

            elif key in ["tau", "q"]:
                data_l = None
                if key == "tau":
                    data_l = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("tau", 0, -1)]
                elif key == "q":
                    data_l = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("q", 0, -1)]

                time_vectors = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("timestamps_cycle", 0, -1)]

                if len(time_vectors) != len(data_l):
                    continue  # timestamps_cycle has not been uploaded to the database yet

                nb_dof = MODEL.nbQ()
                data = np.empty((nb_dof, 0))
                x_data = np.empty((0, ))
                for i_cycle in range(len(data_l)):
                    if data_l[i_cycle] is not None:
                        nb_frames_this_cycle = len(data_l[i_cycle][0])
                        data_this_cycle = np.empty((nb_dof, nb_frames_this_cycle))
                        for i_dof in range(nb_dof):
                            data_this_cycle[i_dof, :] = data_l[i_cycle][i_dof]
                        data = np.concatenate((data, data_this_cycle), axis=1)
                        x_data = np.concatenate((x_data, time_vectors[i_cycle]))

                        if self.initial_time is not None:
                            self.graph_axes[key].plot(
                                np.array([time_vectors[i_cycle][-1] - self.initial_time, time_vectors[i_cycle][-1] - self.initial_time]),
                                np.array([-1000, 1000]),
                                "--",
                                color="black",
                            )

                if key == "q":
                    data = data * 180 / np.pi

                y_data = [data[DOF_CORR["LHip"][0], :],
                          data[DOF_CORR["LAnkle"][0], :],
                          data[DOF_CORR["LKnee"][0], :]]

            elif key == "gait_params" or key == "stim_params" or key == "cost":
                if key == "gait_params":
                    data = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("gait_parameters", 0, -1)]
                elif key == "stim_params":
                    data = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("stimulation_parameters", 0, -1)]
                elif key == "cost":
                    data = [json.loads(x.decode("utf-8")) for x in redis_client.lrange("cost", 0, -1)]

                if data == []:
                    continue

                data = np.array(data)
                y_data = []
                for i_line in range(self.which_data_to_plot[key]["nb_lines"]):
                    y_data += [data[:, i_line]]

                x_data = np.arange(len(y_data[0]))

            else:
                raise RuntimeError("graph key not recognized.")

            if x_data.shape[0] > 0:
                if key in ["forces", "marker", "tau", "q"]:
                    if self.initial_time is None:
                        self.initial_time = x_data[0]
                    x_data -= self.initial_time

                for i_data, data in enumerate(y_data):
                    self.graph_plots[key][i_data].set_xdata(x_data)
                    self.graph_plots[key][i_data].set_ydata(data)
                self.graph_axes[key].set_xlim((x_data[0], x_data[-1]))

            # Draw all the plots now
            self.canvas.draw()

    def create_graphs(self):
        """Updates displayed graphs based on selected checkboxes."""
        self.figure.clear()
        colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:pink"]

        # Check selected graphs
        count = 0
        for key in self.which_data_to_plot.keys():
            is_checked = self.checkboxes_graphs[key].isChecked()
            self.which_data_to_plot[key]["active"] = is_checked
            count += 1 if is_checked else 0

        if count == 0:
            # Nothing to display
            self.canvas.draw()
            return

        # Calculate layout for subplots
        rows = (count + 1) // 2
        cols = 2 if count > 1 else 1
        subplot_index = 1

        # Affichage des graphiques en fonction des cases à cocher
        for key in self.which_data_to_plot.keys():
            is_checked = self.which_data_to_plot[key]["active"]
            if is_checked:
                # Ajouter un sous-graphe pour chaque graphique sélectionné
                ax = self.figure.add_subplot(rows, cols, subplot_index)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(key)
                ax.set_xlim(0, 1)
                linestyle = "-"
                marker = "None"
                if key == "forces":
                    ax.set_ylim(-50, 1000)
                elif key == "marker":
                    ax.set_ylim(0, 2)
                elif key == "tau":
                    ax.set_ylim(-800, 800)
                elif key == "q":
                    ax.set_ylim(-180, 180)
                elif key == "gait_params":
                    ax.set_ylim(0, 2)
                    marker = "o"

                self.graph_axes[key] = ax
                if key not in self.graph_plots:
                    self.graph_plots[key] = [[] for _ in range(self.which_data_to_plot[key]["nb_lines"])]
                for i_plot in range(self.which_data_to_plot[key]["nb_lines"]):
                    self.graph_plots[key][i_plot] = ax.plot(np.array([0, 0]), np.array([0, 0]), linestyle=linestyle, marker=marker, color=colors[i_plot])[0]
                subplot_index += 1

        # Redessiner le canevas pour afficher les nouvelles données
        self.canvas.draw()

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

    # # serveur_virtuel :
    # server_ip = "127.0.0.1"
    # server_port = 50000

    # Main_Bertec_Cometa :
    server_ip = "192.168.0.1"
    server_port = 7

    # Data receiver (goal: interaction with Qualisys)
    data_receiver = DataReceiver(server_ip, server_port)

    # Data processor (goal: ID, IK)
    # data_processor = DataProcessor()
    q_processor = QProcessor()
    # tau_processor = TauProcessor()

    # Stimulation processor (goal: determine if a stim is needed + interaction with stimulator)
    stimulation_processor = StimulationProcessor()

    # Bayesian optimizer (goal: determine which stimulation parameters to try)
    bayesian_optimizer = BayesianOptimizer()

    # --- Thread activation --- #
    threading.Thread(target=data_receiver.start_receiving, daemon=True).start()
    # threading.Thread(target=data_processor.start_processing, daemon=True).start()
    threading.Thread(target=q_processor.start_processing, daemon=True).start()
    # threading.Thread(target=tau_processor.start_processing, daemon=True).start()
    # threading.Thread(target=stimulation_processor.start_processing, daemon=False).start()
    # threading.Thread(target=bayesian_optimizer.start_optimizing, daemon=False).start()

    # Start the GUI
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
