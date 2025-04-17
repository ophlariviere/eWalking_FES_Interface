import biorbd
from scipy.signal import filtfilt, butter, savgol_filter
import numpy as np


class DataProcessor:
    def __init__(self):
        self.cycle_num = 0
        self.dof_corr = {"LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)}

    def calculate_kinematic_dynamic(self, model,force, mks):
        self.calculate_ik(model, mks)

    def calculate_ik(self, model, mks):
        mks = self.fill_missing_markers(mks,5)
        # Marqueurs : (3, n_markers, n_frames)
        freq = 100 #todo adapt auto
        ik = biorbd.InverseKinematics(model, mks)
        ik.solve(method="trf")
        q = ik.q

        # Filtrage (optionnel en temps réel)
        q_filt = savgol_filter(q, 31, 3, axis=1)
        qdot = np.gradient(q_filt, axis=1) * freq
        qddot = np.gradient(qdot, axis=1) * freq
        return q_filt, qdot, qddot

    @staticmethod
    def force_data_filter(data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.empty([len(data[:, 0]), len(data[0, :])])
        for ii in range(3):
            # filtered_data[ii, :] = medfilt(data[ii, :], kernel_size=5)
            filtered_data[ii, :] = filtfilt(b, a, data[ii, :], axis=0)
        return filtered_data

    @staticmethod
    def fill_missing_markers(data, max_interp_gap=10):
        """
        Remplace les NaN dans les données de marqueurs par interpolation ou hold.
        data: np.ndarray de shape (n_markers, 3, n_frames)
        """
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
                    # Le marqueur est complètement absent
                    continue

                for i in range(len(not_nan_idx) - 1):
                    start, end = not_nan_idx[i], not_nan_idx[i + 1]
                    gap = end - start - 1
                    if gap == 0:
                        continue
                    elif gap <= max_interp_gap:
                        # Interpolation linéaire
                        filled[m, d, start + 1:end] = np.linspace(
                            signal[start], signal[end], gap + 2
                        )[1:-1]
                    else:
                        # Trop long : on fait un "hold" (on garde la dernière valeur connue)
                        filled[m, d, start + 1:end] = signal[start]

                # Remplir le début si NaN au début
                first = not_nan_idx[0]
                if first > 0:
                    filled[m, d, :first] = signal[first]

                # Remplir la fin si NaN à la fin
                last = not_nan_idx[-1]
                if last < n_frames - 1:
                    filled[m, d, last + 1:] = signal[last]

        return filled
