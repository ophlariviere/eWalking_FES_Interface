import asyncio
import qtm_rt
from biosiglive import Server
import numpy as np
import xml.etree.ElementTree as ET

#
# class QualisysDataReceiver:
#     def __init__(self, server_ip="192.168.0.1", server_port=7, system_rate=100):
#         self.server = Server(server_ip, server_port)
#         self.server.start()
#         self.server_ip = server_ip
#         self.port = server_port
#         self.interface = None
#         self.system_rate = system_rate
#         self.qualisys_ip = "192.168.254.1"
#         self.target_rate = system_rate  # 100 Hz
#         self.target_period = 1 / self.target_rate  # 10 ms
#
#     async def listen_for_data(self):
#
#         """Établit la connexion avec Qualisys"""
#         self.interface = await qtm_rt.connect(self.qualisys_ip)
#         if self.interface is None:
#             print("Erreur : Impossible de se connecter à Qualisys.")
#             return False
#         print("Connexion établie avec Qualisys.")
#
#         """Écoute les paquets et traite les données"""
#         data_all = {}
#         mks_name = []
#         data_all["mks_name"] = []
#         result = await self.interface.get_parameters(parameters=["3d"])
#         xml = ET.fromstring(result)
#         for idx, label in enumerate(label.text for label in xml.iter("Name")):
#             mks_name += [label]
#         data_all["mks_name"] = mks_name
#
#         while True:
#
#             # Récupère le paquet dès qu'il est disponible
#             packet = await self.interface.get_current_frame(components=["analogsingle", "3d", "force"])
#
#             try:
#                 # extract force data
#                 data_all["force"] = []
#                 _, force_data = packet.get_force()
#                 forces_array = organize_force_data2(force_data)
#                 data_all["force"] = forces_array
#                 print(data_all["force"].shape)
#
#                 # extract mks
#                 data_all["mks"] = []
#                 _, mks_data_n = packet.get_3d_markers()
#                 mks_data = np.array([[p.x, p.y, p.z] for p in mks_data_n])
#                 data_all["mks"] = mks_data
#
#                 # Traite les données analogiques
#                 headers, analog_data = packet.get_analog_single()
#                 data_all["footswitch_data"] = []
#                 if analog_data and (len(analog_data) > 1):
#                     emg_data_all = analog_data[1][1]  # Extraction des données analogiques
#                     if not np.isnan(emg_data_all).any():
#                         print(f"📊 Données analogiques reçues : {emg_data_all}")  # Debugging
#
#                 connection, message = self.server.client_listening()
#                 if connection:
#                     self.server.send_data(data_all, connection, message)
#
#             except Exception as e:
#                 print(e)
#                 continue  # Passe au paquet suivant sans arrêter le programme
#
#             # # Timer pour maintenir la boucle à 100 Hz
#             # elapsed_time = asyncio.get_event_loop().time() - start_time
#             # sleep_time = max(0, self.target_period - elapsed_time)
#             # await asyncio.sleep(sleep_time)
#
# def organize_force_data2(force_data):
#     all_data = []
#     for plate, forces in force_data:
#         plate_data = [[f.x, f.y, f.z, f.x_m, f.y_m, f.z_m, f.x_a, f.y_a, f.z_a] for f in forces]
#         # Transpose pour avoir (9, nb_frames)
#         all_data.append(np.array(plate_data).T)
#     return np.array(all_data)  # shape = (2, 9, nb_frames)
#
#
# if __name__ == "__main__":
#     processor = QualisysDataReceiver(server_ip="192.168.0.1", server_port=7, system_rate=100)
#     asyncio.run(processor.listen_for_data())


QUALISYS_IP = "192.168.254.1"
SERVER_IP = "192.168.0.1"
SERVER_PORT = 7

SYSTEM_RATE = 100

SERVER = Server(SERVER_IP, SERVER_PORT)
SERVER.start()


def format_data(frame_number, header, markers, forces):

    # Header
    data_all = {}
    data_all["frame"] = frame_number
    data_all["header"] = header

    # Organize force data
    force_array = []
    for plate, force in forces:
        plate_data = [[f.x, f.y, f.z, f.x_m, f.y_m, f.z_m, f.x_a, f.y_a, f.z_a] for f in force]
        force_array.append(np.array(plate_data).T)
    data_all["force"] = np.array(force_array) # shape = (2, 9, nb_frames)

    # Organize marker data
    data_all["mks"] = np.array([[p.x, p.y, p.z] for p in markers])

    return data_all


async def send_data_to_server(data_all):
    print("Send data to server")
    connection, message = await SERVER.client_listening()  # TODO FIX THIS
    print("message : ", message)
    if connection:
        SERVER.send_data(data_all, connection, message)
        print("Data sent")


def on_packet(packet):
    """ Callback function that is called everytime a data packet arrives from QTM """
    # print("on paquet")
    frame_number = packet.framenumber
    print(frame_number)
    header, markers = packet.get_3d_markers()
    # print(len(markers))
    _, forces = packet.get_force()
    # print(len(forces))
    data_all = format_data(frame_number, header, markers, forces)
    # print(data_all.keys())
    send_data_to_server(data_all)


async def setup():
    """ Main function """
    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None:
        return

    await connection.stream_frames(components=["3d", "force"], on_packet=on_packet)


if __name__ == "__main__":
    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()