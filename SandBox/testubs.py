import asyncio
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread
from Stim_P24.Stim_Interface import StimInterfaceWidget
from SandBox.Data_Reception import DataReceiver
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    server_ip = ("169.254.171.205",)
    server_port = (7,)

    app = QApplication(sys.argv)

    visualization_widget = StimInterfaceWidget()
    visualization_widget.show()

    data_receiver = DataReceiver(server_ip, server_port, visualization_widget)

    class DataThread(QThread):

        def __init__(self, receiver):
            super().__init__()
            self.receiver = receiver

        def run(self):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.receiver.listen_for_data())

    data_thread = DataThread(data_receiver)
    data_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
