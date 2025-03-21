import asyncio
import qtm_rt
from biosiglive import Server
import numpy as np
import xml.etree.ElementTree as ET


class QualisysDataReceiver:
    def __init__(self, server_ip="192.168.0.1", server_port=7, system_rate=100):
        self.server = Server(server_ip, server_port)
        self.server.start()
        self.server_ip = server_ip
        self.port = server_port
        self.interface = None
        self.system_rate = system_rate
        self.qualisys_ip = "192.168.254.1"
        self.phase_detection_method = "emg"
        self.target_rate = 100  # 100 Hz
        self.target_period = 1 / self.target_rate  # 10 ms

    async def setup(self):
        """Établit la connexion avec Qualisys"""
        self.interface = await qtm_rt.connect(self.qualisys_ip)
        if self.interface is None:
            print("Erreur : Impossible de se connecter à Qualisys.")
            return False
        print("Connexion établie avec Qualisys.")
        return True

    async def listen_for_data(self):
        """Écoute les paquets et traite les données"""
        await self.setup()

        try:
            data_all = {}
            mks_name = []
            data_all["mks_name"] = []
            result = await self.interface.get_parameters(parameters=["3d"])
            xml = ET.fromstring(result)
            for idx, label in enumerate(label.text for label in xml.iter("Name")):
                mks_name += [label]
            data_all["mks_name"] = mks_name

            while True:
                start_time = asyncio.get_event_loop().time()  # Démarre le timer

                # Récupère le paquet dès qu'il est disponible
                packet = await self.interface.get_current_frame(components=["analogsingle", "3d", "force"])

                # Si aucun paquet n'est reçu, passe au suivant
                if not packet:
                    continue

                try:
                    # extract force data
                    data_all["force"] = []
                    _, force_data = packet.get_force()
                    force_data = organize_force_data(force_data)
                    data_all["force"] = force_data

                    # extract mks
                    data_all["mks"] = []
                    _, mks_data = packet.get_3d_markers()
                    data_all["mks"] = mks_data

                    # Traite les données analogiques
                    headers, analog_data = packet.get_analog_single()
                    data_all["footswitch_data"] = []
                    if analog_data and (len(analog_data) > 1):
                        emg_data_all = analog_data[1][1]  # Extraction des données analogiques
                        if not np.isnan(emg_data_all).any():
                            print(f"📊 Données analogiques reçues : {emg_data_all}")  # Debugging

                            if self.phase_detection_method == "emg":
                                if len(emg_data_all) > 0:
                                    data_all["footswitch_data"] = emg_data_all

                    try:
                        connection, message = self.server.client_listening()
                        if connection:
                            self.server.send_data(data_all, connection, message)
                    except Exception as e:
                        print(e)
                        continue

                except Exception as e:
                    continue  # Passe au paquet suivant sans arrêter le programme

                # Timer pour maintenir la boucle à 100 Hz
                elapsed_time = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, self.target_period - elapsed_time)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("⏹️ Arrêt de la boucle de réception des données.")
        except Exception as e:
            print(f"🚨 Erreur inattendue dans la boucle principale : {e}")
        finally:
            print("🔌 Fermeture de la connexion...")
            if self.interface:
                await self.interface.disconnect()


def organize_force_data(forces_data):
    nb_pf = len(forces_data)  # Nombre de plaques de force
    collected_data = []  # Liste dynamique pour collecter les données valides
    nb_frames = max(len(forces_data[0][1]), len(forces_data[1][1]))  # Nombre de frames pour cette plaque
    # Collecte des données
    for plate_num in range(nb_pf):
        PF_Force = forces_data[plate_num][1]
        # Temporaire pour cette plaque
        plate_data = np.empty((9, nb_frames))
        plate_data[:] = np.nan
        for frame_idx, data_tmp in enumerate(PF_Force):
            # Récupérer les données et remplir la matrice temporaire
            plate_data[:, frame_idx] = [
                data_tmp.x,
                data_tmp.y,
                data_tmp.z,
                data_tmp.x_m,
                data_tmp.y_m,
                data_tmp.z_m,
                data_tmp.x_a,
                data_tmp.y_a,
                data_tmp.z_a,
            ]
        collected_data.append(plate_data)

        # Concaténation des données valides uniquement pour obtenir [9 * nb_pf, nb_frame]
    all_forces_data = np.concatenate(collected_data, axis=0)
    return all_forces_data


if __name__ == "__main__":
    processor = QualisysDataReceiver()
    try:
        asyncio.run(processor.listen_for_data())  # Écoute les paquets dès leur arrivée
    except KeyboardInterrupt:
        print("🛑 Arrêt du processus par l'utilisateur.")
