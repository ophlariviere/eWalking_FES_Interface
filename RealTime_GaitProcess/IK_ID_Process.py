import biorbd

class DataProcessor:
    def __init__(self):
        self.cycle_num = 0
        self.dof_corr = {"LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)}

    def calculate_kinematic_dynamic(self, model, force, mks):
        mks=self.fill_missing_markers(mks,5)
        freq = 100  # Hz
        n_frames = next(iter(mks.values())).shape[1]
        marker_names = [n.to_string() for n in model.technicalMarkerNames()]
        # Marqueurs : (3, n_markers, n_frames)

        # Kalman
        kalman = biorbd.KalmanReconsMarkers(model, biorbd.KalmanParam(freq))
        nq, nqdot, nqddot = model.nbQ(), model.nbQdot(), model.nbQddot()
        q_out = np.zeros((nq, n_frames))

        q = biorbd.GeneralizedCoordinates(model)
        qdot = biorbd.GeneralizedVelocity(model)
        qddot = biorbd.GeneralizedAcceleration(model)

        for i in range(n_frames):
            mk = np.reshape(markers_array[:, :, i].T, -1)
            kalman.reconstructFrame(model, mk, q, qdot, qddot)
            q_out[:, i] = q.to_array()

        # Filtrage (optionnel en temps réel)
        q_filt = savgol_filter(q_out, 31, 3, axis=1)
        qdot = np.gradient(q_filt, axis=1) * freq
        qddot = np.gradient(qdot, axis=1) * freq

        # Forces externes
        contact_names = ["LFoot", "RFoot"]
        platform_origin = np.array([
            [0.79165588, 0.77004227, 0.00782072],  # PF1
            [0.7856461, 0.2547548, 0.00760771],  # PF2
        ])

        force = np.zeros((2, 3, n_frames))
        moment = np.zeros((2, 3, n_frames))
        for idx in range(2):
            idx = pf * 9
            force = biorbd.Vector3d(*forces[idx:idx + 3, i])
            moment = biorbd.Vector3d(*forces[idx + 3:idx + 6, i])
            cop = biorbd.Vector3d(*forces[idx + 6:idx + 9, i])


            force[idx] = self.forcedatafilter(force[f"Force_{idx + 1}"], 4, 2000, 10)
            moment[idx] = self.forcedatafilter(force[f"Moment_{idx + 1}"] / 1000, 4, 2000, 10)

        tau_out = np.zeros((nq, n_frames))
        for i in range(n_frames):
            ext_load = model.externalForceSet()
            for idx, contact in enumerate(contact_names):
                F = force[idx, :, i]
                M = moment[idx, :, i]
                if np.linalg.norm(F) > 40:
                    spatial_vec = np.concatenate((M, F))
                    ext_load.add(biorbd.String(contact), spatial_vec, platform_origin[idx])
            tau = model.InverseDynamics(q_filt[:, i], qdot[:, i], qddot[:, i], ext_load)
            tau_out[:, i] = tau.to_array()

        return tau_out, q_filt, qdot, qddot

    def forcedatafilter(data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.empty([len(data[:, 0]), len(data[0, :])])
        for ii in range(3):
            # filtered_data[ii, :] = medfilt(data[ii, :], kernel_size=5)
            filtered_data[ii, :] = filtfilt(b, a, data[ii, :], axis=0)
        return filtered_data

    def fill_missing_markers(self, data, max_interp_gap=10):
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
