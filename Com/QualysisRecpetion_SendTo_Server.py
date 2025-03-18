import asyncio
import qtm_rt
from biosiglive import Server


class QualisysDataReceiver:
    def __init__(
            self,
            server_ip="192.168.0.1",
            server_port=7, system_rate=100):
        self.server = Server(server_ip, server_port)
        self.server.start()
        self.server_ip = server_ip
        self.port = server_port
        self.interface = None
        self.system_rate = system_rate
        self.qualisys_ip = "192.168.254.1"
        self.phase_detection_method = "emg"

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
            while True:
                # Récupère le paquet dès qu'il est disponible
                packet = await self.interface.get_current_frame(components=['analog'])

                # Si aucun paquet n'est reçu, passe au suivant
                if not packet:
                    continue

                try:
                    # Traite les données analogiques

                    headers, analog_data = packet.get_analog()
                    if analog_data and len(analog_data) > 1:
                        analog = analog_data  # Extraction des données analogiques

                        print(f"📊 Données analogiques reçues : {analog}")  # Debugging

                        if self.phase_detection_method == "emg":
                            emg_data = []
                            for device, sample, channel in analog:
                                if hasattr(device, 'id') and (device.id == 2):
                                    emg_data.append(channel[0])

                            if len(emg_data) > 0:
                                try:
                                    connection, message = self.server.client_listening()  # Non-bloquant
                                    if connection:
                                        data = {
                                            "footswitch_data": emg_data,
                                        }
                                        self.server.send_data(data, connection, message)
                                except Exception as e:
                                    print(e)
                                    continue
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
    processor = QualisysDataReceiver()
    try:
        asyncio.run(processor.listen_for_data())  # Écoute les paquets dès leur arrivée
    except KeyboardInterrupt:
        print("🛑 Arrêt du processus par l'utilisateur.")
