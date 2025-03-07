import asyncio
import qtm_rt
from PyQt5.QtCore import   QObject
from FootSwitch.FootSwitchEMGProcess import FootSwitchEMGProcessor


class DataReceiver(QObject):
    def __init__(
            self,
            server_ip,
            server_port,
            visualization_widget,
    ):

        super().__init__()
        self.visualization_widget = visualization_widget

        self.server_ip = server_ip
        self.port = server_port
        self.interface = None  # await qtm_rt.connect(self.server_ip)
        self.foot_switch_emg_processor = FootSwitchEMGProcessor(self.visualization_widget)
        self.emg_pied_droit_heel = 1
        self.emg_pied_droit_toe = 2
        self.emg_pied_gauche_heel = 1
        self.emg_pied_gauche_toe = 2

    def setup(self):
        """Établit la connexion avec Qualisys"""
        self.interface = await qtm_rt.connect(self.server_ip)
        if self.interface is None:
            print("Erreur : Impossible de se connecter à Qualisys.")
            return False
        print("Connexion établie avec Qualisys.")
        return True

    def listen_for_data(self):
        """Écoute les paquets et traite les données"""

        if self.interface is None:
            self.setup()

        try:
            while True:
                packet = await self.interface.get_current_frame(components=['analog'])
                if not packet:
                    continue

                try:
                    _, analog_data = packet.get_analog()
                    if analog_data and len(analog_data) > 1:
                        analog = analog_data

                        # Traitement des données EMG (ajustez ici selon votre logique de détection)
                        emg_data = []
                        for device, sample, channel in analog:
                            if hasattr(device, 'id') and device.id == 2:
                                emg_data.append(channel[0])
                        dict_footswitch = {'footswitch_data': emg_data}
                            except Exception as e:
                                print(f"Erreur EMG: {e}")

                except Exception as e:
                    e=0 # print(f"Erreur de traitement des paquets: {e}")

        except asyncio.CancelledError:
            print("Arrêt de la boucle d'écoute des données.")
        except Exception as e:
            print(f"Erreur inattendue: {e}")
        finally:
            print("Fermeture de la connexion...")
            if self.interface:
                await self.interface.disconnect()

"""
def start_gui():
    app = QApplication(sys.argv)
    gui = StimInterfaceWidget()
    gui.show()

    # Créez l'instance du processeur EMG et du thread pour recevoir les données
    foot_switch_emg_processor = FootSwitchEMGProcessor(gui)
    receiver_thread = DataReceiverThread(foot_switch_emg_processor)

    # Lancez le thread en arrière-plan
    receiver_thread.start()

    # Optionnel: connectez les signaux si vous souhaitez mettre à jour l'interface en fonction des données reçues
    # receiver_thread.data_received.connect(gui.update_data)

    sys.exit(app.exec_())  # Démarre l'application PyQt


if __name__ == "__main__":
    start_gui()
"""