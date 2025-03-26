import numpy as np
import time
import threading
import BertecRemoteControl  # Module de communication avec le tapis
import interface
import scipy.linalg
import zmq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from collections import deque

# ✅ Initialisation de la communication avec le tapis
remote = BertecRemoteControl.RemoteControl()
remote.start_connection()

CENTER_COP = 0.8  # ✅ Centre du tapis à 0.7
dt = 0.01  # Intervalle de temps (10 ms)
COMMAND_DELAY = 0.1  # ✅ Délai entre envois de commandes  --> de base à 0.2
DECELERATION_SMOOTHING = 0.25  # ✅ Facteur de lissage de la décélération --> de base à 0.2


# ✅ Matrices du modèle du COP
A = np.array([[1, dt], [0, 1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])

# ✅ Matrices du LQR
Q = np.diag([30, 10])
R = np.array([[0.05]])
P = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

# ✅ Matrices du filtre de Kalman
Q_kalman = np.diag([0.01, 0.01])
R_kalman = np.array([[0.05]])
P_k = np.eye(2)


class StateEstimator:
    def __init__(self):
        self.X_k = np.array([[CENTER_COP], [0]])
        self.P_k = P_k
        self.force_threshold = 20
        self.fyr = 0
        self.fyl = 0
        self.fzr = 0
        self.fzl = 0


    def read_forces(self):
        force_data = remote.get_force_data()
        if force_data is None:
            print("⚠️ Pas de données reçues, vérifiez la connexion.")
            return 0, CENTER_COP

        fz = force_data.get("fz", 0)
        cop = force_data.get("copy", CENTER_COP)
        self.fyr = force_data.get("fyl", 0) #les plateformes Bertec ont été inversées à l'installation !
        self.fyl = force_data.get("fyr", 0)
        self.fzr = force_data.get("fzl", 0)
        self.fzl = force_data.get("fzr", 0)
        return fz, cop

    def kalman_update(self, cop_measured):
        """Mise à jour du filtre de Kalman"""
        X_k_pred = A @ self.X_k
        P_k_pred = A @ self.P_k @ A.T + Q_kalman

        S_k = C @ P_k_pred @ C.T + R_kalman
        K_kalman = P_k_pred @ C.T @ np.linalg.inv(S_k)
        self.X_k = X_k_pred + K_kalman @ (cop_measured - C @ X_k_pred)
        self.P_k = (np.eye(2) - K_kalman @ C) @ P_k_pred

        return self.X_k

    def update(self):
        fz, cop_measured = self.read_forces()
        X_k = self.kalman_update(cop_measured)

        flag_step = fz > self.force_threshold
        cop_moyen = X_k[0, 0]
        dcom_step = X_k[1, 0]

        return flag_step, cop_moyen, dcom_step, fz


class LQGController:
    def __init__(self, min_v=0.4, max_v=2.0):
        self.min_v = min_v
        self.max_v = max_v
        self.v_tm = min_v
        self.last_command_time = 0

    def compute_target_speed(self, flag_step, cop_moyen, dcom_step, fz):
        """✅ Ajuste immédiatement la vitesse cible en fonction du COP"""
        if not flag_step:
            return self.v_tm

        v_target = 1.0 + 1.5 * (cop_moyen - CENTER_COP) + CENTER_COP * dcom_step

        if fz > 50:
            v_target += 0.15
        elif fz < 25:
            v_target -= 0.1

        v_target = np.clip(v_target, self.min_v, self.max_v)

        # ✅ Applique un lissage uniquement si on ralentit
        if v_target < self.v_tm:
            v_target = self.v_tm * (1 - DECELERATION_SMOOTHING) + v_target * DECELERATION_SMOOTHING

        return v_target

    def update_treadmill_speed(self, v_tm_tgt):
        """✅ Mise à jour fluide du tapis avec délai entre commandes"""
        current_time = time.time()

        if abs(v_tm_tgt - self.v_tm) < 0.01:
            return

        if current_time - self.last_command_time < COMMAND_DELAY:
            return

        try:
            self.v_tm = v_tm_tgt
            remote.run_treadmill(
                f"{self.v_tm:.2f}",
                f"{DECELERATION_SMOOTHING:.2f}",
                f"{DECELERATION_SMOOTHING:.2f}",
                f"{self.v_tm:.2f}",
                f"{DECELERATION_SMOOTHING:.2f}",
                f"{DECELERATION_SMOOTHING:.2f}",
            )
            self.last_command_time = current_time
        except zmq.error.ZMQError as e:
            print(f"⚠️ Erreur ZMQ lors de l'envoi de la commande : {e}")
        except Exception as e:
            print(f"⚠️ Erreur inattendue : {e}")


class TreadmillAIInterface(interface.TreadmillInterface):
    def __init__(self, estimator, controller):
        super().__init__()
        self.estimator = estimator
        self.controller = controller
        self.running = False
        self.step_counter = 0  # Ajout du compteur de pas
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)

    def start(self):
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.running = False
        remote.run_treadmill(0, 0.2, 0.2, 0, 0.2, 0.2)
        #self.speed_label.setText(f"Vitesse actuelle: 0 m/s")

    def run(self):
        while self.running:
            flag_step, cop_moyen, dcom_step, fz = self.estimator.update()

            # Récupération des données brutes sans filtrage
            force_data = remote.get_force_data()
            if force_data:
                copx = force_data.get("copx", 0)  # Utilisation directe des données du tapis
                copy = force_data.get("copy", 0)

                # Mise à jour de l'affichage avec les vraies valeurs
                self.update_cop(copx, copy)

            # Calcul et mise à jour de la vitesse du tapis
            v_tm_tgt = self.controller.compute_target_speed(flag_step, cop_moyen, dcom_step, fz)
            self.controller.update_treadmill_speed(v_tm_tgt)

            treadmill_acceleration = (v_tm_tgt - self.controller.v_tm) / dt
            if flag_step:
                self.step_counter += 1  # Augmente le compteur de pas

            self.log_data(self.step_counter, self.controller.v_tm, treadmill_acceleration, copy, cop_moyen)

            self.speed_label.setText(f"Vitesse actuelle: {self.controller.v_tm:.2f} m/s")
            self.cop_x_label.setText(f"COP X : {copx:.2f} m")
            self.cop_y_label.setText(f"COP Y : {copy:.2f} m")

            time.sleep(0.01)

class PropulsionDetector(threading.Thread):
    def __init__(self, estimator):
        super().__init__(daemon=True)
        self.estimator = estimator
        self.running = True

        # Buffers pour les forces
        self.force_buffer_right = []
        self.force_buffer_left = []

        # États de phase
        self.phase_right = "idle"
        self.phase_left = "idle"
        self.plotter = None  # facultatif, assigné ensuite
        self.sendStim = {'right': False, 'left': False}

    def stop(self):
        self.running = False

    def run(self):
        len_buffer = 30
        buffer = deque(maxlen=len_buffer)
        while self.running:
            force_data_on_frame = [self.estimator.fyr, self.estimator.fzr, self.estimator.fyl, self.estimator.fzl]
            buffer.append(force_data_on_frame)

            if len(buffer) == len_buffer:
                data = np.array(buffer)
                self.detect_phase("right", data[:, 0], data[:, 1])
                self.detect_phase("left", data[:, 2], data[:, 3])
            time.sleep(dt)  # 10 ms

    def detect_phase(self, side, force_value_y, force_value_z):
        mass = 500
        fs = 1/dt
        b, a = butter(2, 10 / (0.5 * fs), btype='low')
        force_ap_filter = filtfilt(b, a, force_value_y)
        force_vert_filter = filtfilt(b, a, force_value_z)
        force_vert_last = force_vert_filter[-1:]

        if np.abs(force_vert_last) > 0.3*mass:

            if force_ap_filter[-2] > force_ap_filter[-1] and self.sendStim[side] is False:
                self.sendStim[side] = True
                if side == "right":
                    self.phase_right = "propulsion"
                    print("➡️ Heel-off détecté jambe DROITE")
                    if self.plotter:
                        self.plotter.add_trigger("right", "heel-off")
                else:
                    self.phase_left = "propulsion"
                    print("⬅️ Heel-off détecté jambe GAUCHE")
                    if self.plotter:
                        self.plotter.add_trigger("left", "heel-off")

        if (np.abs(force_vert_filter[-1]) < 0.1*mass) and self.sendStim[side] is True:
            self.sendStim[side] = False
            if side == "right":
                self.phase_right = "fin"
                print("➡️ Toe-off détecté jambe DROITE")
                if self.plotter:
                    self.plotter.add_trigger("right", "toe-off")
            else:
                self.phase_left = "fin"
                print("⬅️ Toe-off détecté jambe GAUCHE")
                if self.plotter:
                    self.plotter.add_trigger("left", "toe-off")


class ForcePlotter:
    def __init__(self, estimator, detector, max_points=500):
        self.estimator = estimator
        self.detector = detector
        self.max_points = max_points

        self.fyr_data = []
        self.fyl_data = []
        self.triggers_right = []
        self.triggers_left = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.line_fyr, = self.ax1.plot([], [], label="FY Right", color="blue")
        self.line_fyl, = self.ax2.plot([], [], label="FY Left", color="green")

        self.texts_right = []
        self.texts_left = []

        self.ax1.set_title("Jambe Droite - FYR")
        self.ax2.set_title("Jambe Gauche - FYL")

        for ax in [self.ax1, self.ax2]:
            ax.set_ylim(-200, 200)
            ax.set_xlim(0, self.max_points)
            ax.grid(True)
            ax.legend()

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=10)
        plt.tight_layout()
        plt.show(block=False)

    def add_trigger(self, side, label):
        index = len(self.fyr_data)
        if side == "right":
            self.triggers_right.append((index, label))
        else:
            self.triggers_left.append((index, label))

    def update_plot(self, frame):
        # Ajoute les nouvelles valeurs
        self.fyr_data.append(self.estimator.fyr)
        self.fyl_data.append(self.estimator.fyl)

        # Garde une longueur fixe
        if len(self.fyr_data) > self.max_points:
            self.fyr_data = self.fyr_data[-self.max_points:]
            self.fyl_data = self.fyl_data[-self.max_points:]
            self.triggers_right = [(i - 1, t) for i, t in self.triggers_right if i >= len(self.fyr_data) - self.max_points]
            self.triggers_left = [(i - 1, t) for i, t in self.triggers_left if i >= len(self.fyl_data) - self.max_points]

        x_vals = list(range(len(self.fyr_data)))

        self.line_fyr.set_data(x_vals, self.fyr_data)
        self.line_fyl.set_data(x_vals, self.fyl_data)

        self.ax1.set_xlim(max(0, len(self.fyr_data) - self.max_points), len(self.fyr_data))
        self.ax2.set_xlim(max(0, len(self.fyl_data) - self.max_points), len(self.fyl_data))

        # Nettoyer les anciens textes
        for t in self.texts_right + self.texts_left:
            t.remove()
        self.texts_right.clear()
        self.texts_left.clear()

        # Afficher les triggers
        for i, label in self.triggers_right:
            if i >= len(x_vals): continue
            y = self.fyr_data[i]
            p = self.ax1.plot(i, y, "ro")[0]
            txt = self.ax1.text(i, y + 5, label, color="red", fontsize=8)
            self.texts_right.append(p)
            self.texts_right.append(txt)

        for i, label in self.triggers_left:
            if i >= len(x_vals): continue
            y = self.fyl_data[i]
            p = self.ax2.plot(i, y, "ro")[0]
            txt = self.ax2.text(i, y + 5, label, color="red", fontsize=8)
            self.texts_left.append(p)
            self.texts_left.append(txt)

        return self.line_fyr, self.line_fyl


if __name__ == "__main__":
    app = interface.QApplication([])
    estimator = StateEstimator()
    controller = LQGController()
    gui = TreadmillAIInterface(estimator, controller)

    propulsion_detector = PropulsionDetector(estimator)
    propulsion_detector.start()

    plotter = ForcePlotter(estimator, propulsion_detector)
    propulsion_detector.plotter = plotter

    gui.show()
    app.exec_()


