import asyncio
import qtm_rt
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from FootSwitch.FootSwitchEMGProcess import FootSwitchEMGProcessor
from Stim_P24.Stim_Interface import StimInterfaceWidget


class DataReceiverThread(QThread):
    # Signal pour envoyer les données traitées à l'interface principale
    data_received = pyqtSignal(list)

    def __init__(self, foot_switch_emg_processor, server_ip="169.254.171.205", port=7):
        super().__init__()
        self.processor = DataReceiver(foot_switch_emg_processor, server_ip, port)

    def run(self):
        """ Exécute la boucle d'écoute des données dans un thread séparé """
        asyncio.run(self.processor.listen_for_data())
        # Si vous devez envoyer des données traitées à l'interface graphique, vous pouvez émettre un signal.
        # self.data_received.emit(data)


class DataReceiver:
    def __init__(self, foot_switch_emg_processor, server_ip="169.254.171.205", port=7):
        self.server_ip = server_ip
        self.port = port
        self.foot_switch_emg_processor = foot_switch_emg_processor
        self.interface = None
        self.emg_pied_droit_heel = 1
        self.emg_pied_droit_toe = 2
        self.emg_pied_gauche_heel = 1
        self.emg_pied_gauche_toe = 2

    async def setup(self):
        """Établit la connexion avec Qualisys"""
        self.interface = await qtm_rt.connect(self.server_ip)
        if self.interface is None:
            print("Erreur : Impossible de se connecter à Qualisys.")
            return False
        print("Connexion établie avec Qualisys.")
        return True

    async def listen_for_data(self):
        """Écoute les paquets et traite les données"""
        if not await self.setup():
            return  # Arrête l'exécution si la connexion échoue

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

                        if len(emg_data) > 0:
                            try:
                                # Remplacez ici par la méthode qui traite vos données EMG
                                self.foot_switch_emg_processor.heel_off_detection(emg_data[self.emg_pied_droit_heel-1],
                                                                                  emg_data[self.emg_pied_droit_toe-1],
                                                                                  1)
                            except Exception as e:
                                print(f"Erreur EMG: {e}")

                except Exception as e:
                    a=0 # print(f"Erreur de traitement des paquets: {e}")

        except asyncio.CancelledError:
            print("Arrêt de la boucle d'écoute des données.")
        except Exception as e:
            print(f"Erreur inattendue: {e}")
        finally:
            print("Fermeture de la connexion...")
            if self.interface:
                await self.interface.disconnect()


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
