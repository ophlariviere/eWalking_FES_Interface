import asyncio
import qtm_rt
from biosiglive import Server
import numpy as np
import xml.etree.ElementTree as ET
import datetime


QUALISYS_IP = "192.168.254.1"
SERVER_IP = "192.168.0.1"
SERVER_PORT = 7

SYSTEM_RATE = 100
MARKER_NAMES = []
NUMBER_OF_FORCE_DATA = 0
TIC = 0

SERVER = Server(SERVER_IP, SERVER_PORT)
SERVER.start()


def format_data(frame_number, header, markers, forces):
    global MARKER_NAMES

    # Header
    data_all = {}
    data_all["frame"] = frame_number
    data_all["header"] = header
    data_all["mks_name"] = MARKER_NAMES

    # Organize force data
    force_array = []
    for plate, force in forces:
        plate_data = [[f.x, f.y, f.z, f.x_m, f.y_m, f.z_m, f.x_a, f.y_a, f.z_a] for f in force]
        force_array.append(np.array(plate_data).T)
    data_all["force"] = np.array(force_array) # shape = (2, 9, nb_frames)

    # Organize marker data
    data_all["mks"] = np.array([[p.x, p.y, p.z] for p in markers])

    return data_all


def send_data_to_server(data_all):
    # print("Send data to server")
    connection, message = SERVER.client_listening()  # If the client (other computer) is not listening, this is blocking
    # print("message : ", message)
    if connection:
        SERVER.send_data(data_all, connection, message)
        # print("Data sent")


def on_packet(packet):
    """ Callback function that is called everytime a data packet arrives from QTM """
    global NUMBER_OF_FORCE_DATA, TIC

    # print("on paquet")
    frame_number = packet.framenumber
    if frame_number % 1000 == 0:
        TOC = datetime.datetime.timestamp(datetime.datetime.now())
        elapsed_time = TOC - TIC
        print(frame_number, " : ", elapsed_time, "  -----  ", NUMBER_OF_FORCE_DATA / elapsed_time, " Hz")
        TIC = TOC
        NUMBER_OF_FORCE_DATA = 0

    header, markers = packet.get_3d_markers()
    # print(len(markers))
    _, forces = packet.get_force()
    # print(len(forces))
    if forces[0][0].force_number != 0:  # TODO: see for markers
        data_all = format_data(frame_number, header, markers, forces)
        # print(data_all.keys())
        NUMBER_OF_FORCE_DATA += data_all["force"].shape[2]
        send_data_to_server(data_all)
        # print(np.nanmax(data_all["force"], axis=1))


async def setup():
    """ Main function """
    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None:
        return

    global MARKER_NAMES
    parameters = await connection.get_parameters(parameters=["3d"])
    xml = ET.fromstring(parameters)
    mks_name = []
    for idx, label in enumerate(label.text for label in xml.iter("Name")):
        mks_name += [label]
    MARKER_NAMES = mks_name

    await connection.stream_frames(components=["3d", "force"], on_packet=on_packet)


if __name__ == "__main__":
    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()