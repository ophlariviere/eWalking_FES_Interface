"""
This version needs
- pip install redis (once)
- redis-server (every time the code is run to start the redis server)
"""

import sys
from PyQt5.QtWidgets import QApplication
from Stim_P24.Stim_Interface import StimInterfaceWidget
from Received_data_force import DataReceiver
import logging
from PyQt5.QtCore import QThread
import redis
import threading


# Configure le logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class Interface(QThread):
    def __init__(self, server_ip, server_port, visualization_widget):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.visualization_widget = visualization_widget

    def run(self):
        # Do stuff

        # Push the updated values for the stimulation parameters to the Redis database
        r.lpush("stimulation_parameters", stimulation_parameters)
        r.ltrim("stimulation_parameters", 0, BUFFER_LENGTH - 1)

        # Get the data to plot
        r.lrange("force", start=0, stop=-1)
        r.lrange("q", start=0, stop=-1)
        r.lrange("tau", start=0, stop=-1)


class DataReceiver:
    def __init__(self, server_ip, server_port, read_frequency=100):
        self.server_ip = server_ip
        self.server_port = server_port
        self.read_frequency = read_frequency
        # ... Do stuff to initialize

    def start_receiving(self):
        received_data = self.tcp_client.get_data_from_server(command=["force", "mks", "mks_name"])
        # Append a list of data (one list item per frame)
        r.lpush("force", received_data["force"])  # Push data to the list
        r.ltrim("force", 0, BUFFER_LENGTH - 1)  # Trim the list to keep only the latest BUFFER_LENGTH items
        r.lpush("mks", received_data["mks"])
        r.ltrim("mks", 0, BUFFER_LENGTH - 1)
        r.lpush("mks_name", received_data["mks_name"])
        r.ltrim("mks_name", 0, BUFFER_LENGTH - 1)


class DataProcessor:
    def __init__(self):
        print("Do stuff to initialize if needed")

    def start_processing(self):
        """
        Use start=0, stop=-1 for the whole buffer (stop is inclusive)
        Use start=-1, stop=-1 for the last data added to the database
        """
        r.lrange("force", start=0, stop=-1)
        r.lrange("mks", start=0, stop=-1)
        r.lrange("mks_name", start=0, stop=-1)
        # The data is a list of np vectors, so we probably need to concatenate the frames first to gat a np matrix

        # Determine when to trim to get a complete cycle
        # If it is a new cycle, ID/IK

        # Put the result in the database
        r.lpush("q", q)
        r.ltrim("q", 0, q.shape[2])  # Trim so that we only keep the data from this cycle
        r.lpush("tau", q)
        r.ltrim("tau", 0, tau.shape[2])

class StimulationProcessor:
    def __init__(self):
        print("Do stuff to initialize if needed")

    def start_processing(self):
        r.lrange("force", start=0, stop=-1)
        r.lrange("stimulation_parameters", start=-1, stop=-1)
        # The stimulation_parameters are updated both by the Interface and the BayesianOptimizationProcessor
        # Determine if a stim is needed
        # If yes, stimulate


def main():
    # --- Server communication parameters --- #
    server_ip = "192.168.0.1"
    server_port = 7

    # Define buffer length
    BUFFER_LENGTH = 1000

    # --- Connect and initialize to Redis server --- #
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # Initialize with empty values
    r.lpush("force", [None for _ in range(BUFFER_LENGTH)])
    r.lpush("mks", [None for _ in range(BUFFER_LENGTH)])
    r.lpush("mks_name", [None for _ in range(BUFFER_LENGTH)])
    r.lpush("q", [None for _ in range(BUFFER_LENGTH)])
    r.lpush("tau", [None for _ in range(BUFFER_LENGTH)])
    r.lpush("stimulation_parameters", [None for _ in range(BUFFER_LENGTH)])


    # ---  Thread definitions --- #

    # GUI (goal: interaction with the user)
    app = QApplication(sys.argv)
    visualization_widget = StimInterfaceWidget()
    visualization_widget.show()

    # Data receiver (goal: interaction with Qualisys)
    data_receiver = DataReceiver(server_ip, server_port)

    # Data processor (goal: ID, IK)
    data_processor = DataProcessor()

    # Stimulation processor (goal: determine if a stim is needed + interaction with stimulator)
    stimulation_processor = StimulationProcessor()


    # --- Thread activation --- #
    Interface(server_ip, server_port, visualization_widget).start()  # Start the interface on its own thread
    threading.Thread(target=data_receiver.start_receiving()).start()
    threading.Thread(target=data_processor.start_processing()).start()
    threading.Thread(target=stimulation_processor.start_processing()).start()


    # --- Start the GUI --- #
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()