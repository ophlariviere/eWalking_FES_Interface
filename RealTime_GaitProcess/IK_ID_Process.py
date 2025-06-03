import biorbd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


class DataProcessor:
    def __init__(self):
        self.cycle_num = 0
        self.dof_corr = {
            "LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
            "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
            "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
            "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
            "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)
        }

    def calculate_kinematic_dynamic(self, model, force, mks):
        q_filt, qdot, qddot = self.calculate_ik(model, mks)
        self.calculate_id(model, force, q_filt, qdot, qddot)
        self.cycle_num = self.cycle_num + 1

    def calculate_ik(self, model, mks):
        mks = self.fill_missing_markers(mks, 5)
        freq = 100  # TODO: adapt frequency dynamically if needed
        ik = biorbd.InverseKinematics(model, mks)
        ik.solve(method="trf")
        q = ik.q

        q_filt = savgol_filter(q, 31, 3, axis=1)
        qdot = np.gradient(q_filt, axis=1) * freq
        qddot = np.gradient(qdot, axis=1) * freq
        return q_filt, qdot, qddot

    def calculate_id(self, model, force, q, qdot, qddot, platform_origin=None):
        num_contacts = len(force)
        num_frames = force[0].shape[1]

        if platform_origin is None:
            platform_origin = [[0, 0, 0] for _ in range(num_contacts)]

        fs_pf = 2000
        fs_mks = 100
        sampling_factor = int(fs_pf / fs_mks)

        force_filtered = np.zeros((num_contacts, 3, num_frames))
        moment_filtered = np.zeros((num_contacts, 3, num_frames))
        tau_data = np.zeros((model.nbQ(), num_frames))

        for contact_idx in range(num_contacts):
            force_filtered[contact_idx] = self.data_filter(force[contact_idx][0:3], 2, fs_mks, 10)
            moment_filtered[contact_idx] = self.data_filter(force[contact_idx][3:6], 4, fs_mks, 10)

        for i in range(num_frames):
            ext_load = model.externalForceSet()
            for contact_idx in range(num_contacts):
                fz = force_filtered[contact_idx, 2, i]
                if fz > 30:
                    force_vec = force_filtered[contact_idx, :, i]
                    moment_vec = moment_filtered[contact_idx, :, i]
                    spatial_vector = np.concatenate((moment_vec, force_vec))
                    point_app = platform_origin[contact_idx]
                    segment_name = "LFoot" if contact_idx == 0 else "RFoot"
                    ext_load.add(biorbd.String(segment_name), spatial_vector, point_app)

            tau = model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], ext_load)
            tau_data[:, i] = tau.to_array()

        return tau_data

    @staticmethod
    def fill_missing_markers(data, max_interp_gap=10):
        filled = data.copy()
        n_markers, _, n_frames = data.shape

        for m in range(n_markers):
            for d in range(3):
                signal = filled[m, d, :]
                nans = np.isnan(signal)
                if not np.any(nans):
                    continue

                not_nan_idx = np.flatnonzero(~nans)
                if len(not_nan_idx) == 0:
                    continue

                for i in range(len(not_nan_idx) - 1):
                    start, end = not_nan_idx[i], not_nan_idx[i + 1]
                    gap = end - start - 1
                    if gap == 0:
                        continue
                    elif gap <= max_interp_gap:
                        filled[m, d, start + 1:end] = np.linspace(signal[start], signal[end], gap + 2)[1:-1]
                    else:
                        filled[m, d, start + 1:end] = signal[start]

                filled[m, d, :not_nan_idx[0]] = signal[not_nan_idx[0]]
                filled[m, d, not_nan_idx[-1] + 1:] = signal[not_nan_idx[-1]]

        return filled

    def data_filter(self, data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low')

        data = np.asarray(data)
        filtered_data = np.empty_like(data)

        if data.ndim == 2:  # (3, T)
            for i in range(3):
                filtered_data[i, :] = self.nan_filtfilt(b, a, data[i, :])
        elif data.ndim == 3:  # (3, N, T)
            for i in range(3):
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
