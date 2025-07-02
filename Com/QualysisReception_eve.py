import asyncio
import qtm_rt
from biosiglive import Server
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from time import sleep
import socket


def test_server_connection():
    global SERVER, SERVER_PORT, SERVER_IP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        try:
            s.connect((SERVER_IP, SERVER_PORT))
        except Exception as e:
            print("Failed to connect: ", e)

        print("Connexion successful to data streaming server.")


QUALISYS_IP = "192.168.254.1"
SERVER_IP = "192.168.0.1"
SERVER_PORT = 7

SYSTEM_RATE = 100
MARKER_NAMES = []
NUMBER_OF_FORCE_DATA = 0
TIC_FORCE_DATA = 0
FRAME_COUNTER = 0
TIC_MARKER_DATA = 0


SERVER = Server(SERVER_IP, SERVER_PORT)
SERVER.start()
test_server_connection()
sleep(1)
ACQUISITION_RATE = SERVER.acquisition_rate
print("Acquisition rate : ", SERVER.acquisition_rate)


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
    data_all["force"] = np.array(force_array)  # shape = (2, 9, nb_frames)

    # Organize marker data
    data_all["mks"] = np.array([[p.x, p.y, p.z] for p in markers]) / 1000

    data_all["timestamp"] = datetime.timestamp(datetime.now())

    return data_all


def send_data_to_server(data_all):
    connection, message = SERVER.client_listening()  # If the client (other computer) is not listening, this is blocking
    if connection:
        SERVER.send_data(data_all, connection, message)


def on_packet(packet):
    """ Callback function that is called everytime a data packet arrives from QTM """
    global NUMBER_OF_FORCE_DATA, TIC_FORCE_DATA, TIC_MARKER_DATA, FRAME_COUNTER

    PRINT_FREQUENCY_FLAG = True

    # Get the data
    frame_number = packet.framenumber
    header, markers = packet.get_3d_markers()
    _, forces = packet.get_force()

    # Format data in a readable way
    data_all = format_data(frame_number, header, markers, forces)

    if forces[0][0].force_number != 0:
        NUMBER_OF_FORCE_DATA += data_all["force"].shape[2]

    # Print the frequency at which the data is sent to the TCP server
    if PRINT_FREQUENCY_FLAG:
        if frame_number % 1000 == 0:
            TOC = datetime.timestamp(datetime.now())
            elapsed_time = TOC - TIC_FORCE_DATA
            print(frame_number, " : ", elapsed_time, "  -----  ", NUMBER_OF_FORCE_DATA / elapsed_time, " Hz")
            TIC_FORCE_DATA = TOC
            NUMBER_OF_FORCE_DATA = 0

        FRAME_COUNTER += 1
        if FRAME_COUNTER % 100 == 0:
            TOC = datetime.timestamp(datetime.now())
            elapsed_time = TOC - TIC_MARKER_DATA
            print(" Markers  -----  ", FRAME_COUNTER / elapsed_time, " Hz")
            TIC_MARKER_DATA = TOC
            FRAME_COUNTER = 0

    # Actually send the data to the TCP server
    send_data_to_server(data_all)

async def get_marker_names(connection):
    global MARKER_NAMES
    parameters = await connection.get_parameters(parameters=["3d"])
    xml = ET.fromstring(parameters)
    mks_name = []
    for idx, label in enumerate(label.text for label in xml.iter("Name")):
        mks_name += [label]
    MARKER_NAMES = mks_name
    if len(mks_name) != 16:
        raise RuntimeError("The model specified in Qualisys is not reduced_marketset_lower_body")

async def setup_stream_frames():
    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None:
        return

    await get_marker_names(connection)

    await connection.stream_frames(components=["3d", "force"], on_packet=on_packet)


async def setup_get_current_frame():
    global ACQUISITION_RATE

    print("setup_get_current_frame")

    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None:
        print("No connection")
        return

    await get_marker_names(connection)

    start_time = asyncio.get_event_loop().time()  # Démarre le timer
    while True:

        # Récupère le paquet dès qu'il est disponible
        packet = await connection.get_current_frame(components=["3d", "force"])

        # Si aucun paquet n'est reçu, passe au suivant
        if not packet:
            continue

        try:
            on_packet(packet)

            # Timer pour maintenir la boucle à 100 Hz
            elapsed_time = asyncio.get_event_loop().time() - start_time
            sleep_time = 1 / ACQUISITION_RATE - elapsed_time
            if sleep_time < 0:
                raise RuntimeError("The code is too slow for the acquisition rate")
            await asyncio.sleep(sleep_time)

            start_time = asyncio.get_event_loop().time()  # Démarre le timer
        except:
            continue  # Passe au paquet suivant sans arrêter le programme


if __name__ == "__main__":


    # MODE = "stream_frames"
    # asyncio.ensure_future(setup_stream_frames())
    # asyncio.get_event_loop().run_forever()

    MODE = "get_current_frame"
    asyncio.run(setup_get_current_frame())
