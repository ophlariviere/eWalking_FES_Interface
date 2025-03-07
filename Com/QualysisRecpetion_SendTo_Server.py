import asyncio
import time
import qtm_rt
from biosiglive import Server


class QualisysDataReceiver:
    def __init__(
            self,
            server_ip="169.254.171.205",
            server_port=7, system_rate=100):
        self.server = Server(server_ip, server_port)
        self.server.start()
        self.server_ip = server_ip
        self.port = server_port
        self.interface = None
        self.system_rate = system_rate
        self.qualisys_ip = "192.168.254.1"

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
                tic = asyncio.get_event_loop().time()
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

                connection, message = self.server.client_listening()  # Non-bloquant
                if connection:
                    if emg_data is not []:
                        self.server.send_data(dict_footswitch, connection, message)

                loop_time = time.perf_counter() - tic
                real_time_to_sleep = (1 / self.system_rate) - loop_time
                if real_time_to_sleep > 0:
                    await asyncio.sleep(real_time_to_sleep)

        except asyncio.CancelledError:
            print("Arrêt de la boucle d'écoute des données.")
        except Exception as e:
            print(f"Erreur inattendue: {e}")
        finally:
            print("Fermeture de la connexion...")
            if self.interface:
                self.interface.disconnect()


if __name__ == "__main__":
    processor = QualisysDataReceiver()
    asyncio.run(processor.listen_for_data())
