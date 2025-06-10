from custom_interface import MyInterface
from biosiglive import load, RealTimeProcessingMethod, InterfaceType, DeviceType, Server, InverseKinematicsMethods
import numpy as np
import time


# Fonction pour détecter si Fz dépasse le seuil
def detect_start(previous_f_z, current_f_z, threshold=30):
    # Détection du passage de inférieur à supérieur au seuil
    return previous_f_z <= threshold < current_f_z


class RealTimeDataProcessor:
    def __init__(self, server_ip="127.0.0.1", port=50000, data_path="example\\walkAll_LAO01_Cond10.bio",
                 model_path="example\\LAO.bioMod",
                 threshold=30, system_rate=100, device_rate=2000, nb_markers=49, nb_seconds=60):
        # Initialisation du serveur
        self.server = Server(server_ip, port)
        self.server.start()

        # Initialisation de l'interface
        self.interface = MyInterface(system_rate=system_rate, data_path=data_path)
        self.model_path = model_path

        # Paramètres
        self.threshold = threshold
        self.system_rate = system_rate
        self.device_rate = device_rate
        self.nb_markers = nb_markers
        self.nb_seconds = nb_seconds

        # Variables d'état
        self.sending_started = False
        self.previous_fz = 0  # Valeur initiale de Fz

        # Chargement des noms des marqueurs
        self.mks_name = self.load_marker_names()

        # Configuration de l'interface
        self.setup_interface()

    def load_marker_names(self):
        # Chargement des noms des marqueurs à partir du fichier
        tmp = load("example\\walkAll_LAO01_Cond10.bio")
        return tmp['markers_names'].data[0:self.nb_markers].tolist()

    def setup_interface(self):
        # Configuration du jeu de marqueurs
        self.interface.add_marker_set(
            nb_markers=self.nb_markers,
            data_buffer_size=self.system_rate * self.nb_seconds,
            processing_window=self.system_rate * self.nb_seconds,
            marker_data_file_key="markers",
            name="markers",
            rate=self.system_rate,
            kinematics_method=InverseKinematicsMethods.BiorbdKalman,
            model_path=self.model_path,
            unit="mm",
        )

        # Configuration du dispositif (tapis roulant)
        self.interface.add_device(
            18,
            name="Treadmill",
            device_type=DeviceType.Generic,
            rate=self.device_rate,
            data_buffer_size=int(self.device_rate * self.nb_seconds),
            processing_window=int(self.device_rate * self.nb_seconds),
            device_data_file_key="treadmill",
        )

    def process_data(self):
        try:
            while True:
                tic = time.perf_counter()
                dataforce_ok =[[],[]]
                dataforce = self.interface.get_device_data(device_name="Treadmill")
                dataforce_ok[0] = dataforce[0:9 , :]
                dataforce_ok[1] = dataforce[9:, :]
                #Q, _, mark_tmp = self.interface.get_kinematics_from_markers(marker_set_name="markers", get_markers_data=True)
                mark_tmp, _ = self.interface.get_marker_set_data()
                marker = np.transpose(mark_tmp,(1,0,2))
                marker=np.squeeze(marker, axis=-1)
                # Calcul de la force verticale moyenne actuelle
                current_fz = np.mean(dataforce[2])

                if not self.sending_started and detect_start(self.previous_fz, current_fz, self.threshold):
                    self.sending_started = True
                    print("Démarrage de l'envoi des données.")

                elif self.sending_started:
                    connection, message = self.server.client_listening()  # Non-bloquant
                    if connection:
                        dataAll = {
                            "force": dataforce_ok,
                            "mks": marker,
                            "mks_name": self.mks_name
                        }
                        #"Angle": Q[:, -1],
                        self.server.send_data(dataAll, connection, message)
        except KeyboardInterrupt:
            print("Arrêt manuel du programme.")
        except Exception as e:
            print(f"Erreur rencontrée : {e}")
        finally:
            self.server.stop()
            print("Serveur arrêté proprement.")


if __name__ == "__main__":
    processor = RealTimeDataProcessor()
    processor.process_data()
