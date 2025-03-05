import asyncio
import qtm_rt
from FootSwitch.FootSwitchDataProcess import FootSwitchDataProcessor
from FootSwitch.FootSwitchEMGProcess import FootSwitchEMGProcessor

foot_switch_processor = FootSwitchDataProcessor()
foot_switch_emg_processor = FootSwitchEMGProcessor()


class DataReceiver:
    def __init__(self, server_ip="169.254.171.205", port=7, system_rate=100, device_rate=1000):
        # Paramètres
        self.server_ip = server_ip
        self.port = port
        self.system_rate = system_rate
        self.device_rate = device_rate
        self.interface = None
        self.emg_pied_gauche_heel = 1
        self.emg_pied_gauche_toe = 2
        self.emg_pied_droit_heel = 1
        self.emg_pied_droit_toe = 2
        self.analog_channel_foot_switch = 16
        self.phase_detection_method = 'emg'

    async def setup(self):
        """Établit la connexion avec Qualisys"""
        self.interface = await qtm_rt.connect(self.server_ip)
        if self.interface is None:
            print("Erreur : Impossible de se connecter à Qualisys.")
            return False
        print("Connexion établie avec Qualisys.")
        return True

    async def listen_for_data(self):
        """Boucle qui écoute les paquets dès leur arrivée"""
        if not await self.setup():
            return  # Arrête l'exécution si la connexion échoue

        try:
            while True:
                # Récupère le paquet dès qu'il est disponible
                packet = await self.interface.get_current_frame(components=['analog'])

                # Si aucun paquet n'est reçu, passe au suivant
                if not packet:
                    continue

                try:
                    # Traite les données analogiques
                    _, analog_data = packet.get_analog()
                    if analog_data and len(analog_data) > 1:
                        analog = analog_data  # Extraction des données analogiques

                        # print(f"📊 Données analogiques reçues : {analog}")  # Debugging

                        if self.phase_detection_method == "analog":
                            foot_switch_canal = self.analog_channel_foot_switch - 1
                            try:
                                foot_switch_processor.gait_phase_detection(
                                    foot_switch_data=analog[1][foot_switch_canal][2][0]
                                )
                            except IndexError as e:
                                print(f"❌ Erreur d'indexation dans les données analogiques : {e}")

                        elif self.phase_detection_method == "emg":
                            emg_data = []
                            for device, sample, channel in analog:
                                if hasattr(device, 'id') and device.id == 2:
                                    emg_data.append(channel[0])

                            if len(emg_data) > 0:
                                try:
                                    foot_switch_emg_processor.heel_off_detection(
                                        emg_data[self.emg_pied_droit_heel - 1],
                                        emg_data[self.emg_pied_droit_toe - 1],
                                        1
                                    )
                                except IndexError as e:
                                    print(f"❌ Erreur d'indexation dans le traitement EMG : {e}")

                except Exception as e:
                    # Si une erreur survient dans le traitement du paquet, on affiche l'erreur et on passe au paquet suivant
                    # print(f"⚠️ Erreur lors du traitement du paquet : {e}")
                    continue  # Passe au paquet suivant sans arrêter le programme

        except asyncio.CancelledError:
            print("⏹️ Arrêt de la boucle de réception des données.")
        except Exception as e:
            print(f"🚨 Erreur inattendue dans la boucle principale : {e}")
        finally:
            print("🔌 Fermeture de la connexion...")
            if self.interface:
                await self.interface.disconnect()


if __name__ == "__main__":
    processor = DataReceiver()
    try:
        asyncio.run(processor.listen_for_data())  # Écoute les paquets dès leur arrivée
    except KeyboardInterrupt:
        print("🛑 Arrêt du processus par l'utilisateur.")
