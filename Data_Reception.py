import asyncio
import qtm_rt
import time
from FootSwitch.FootSwitchDataProcess import FootSwitchDataProcessor
from FootSwitch.FootSwitchEMGProcess import FootSwitchEMGProcessor
foot_switch_processor = FootSwitchDataProcessor()
foot_switch_emg_processor = FootSwitchEMGProcessor()


class DataReceiver:
    def __init__(self, server_ip="192.168.254.1", port=7, system_rate=100, device_rate=2000):
        # Paramètres
        self.server_ip = server_ip
        self.port = port
        self.system_rate = system_rate
        self.device_rate = device_rate
        self.interface = None
        self.emg_pied_gauche = 1
        self.emg_pied_droit = 2
        self.analog_channel_footswitch = 16
        self.phase_detection_method = 'emg'

    async def setup(self):
        """Setup connection to QTM"""
        self.interface = await qtm_rt.connect(self.server_ip)
        if self.interface is None:
            return
        print("Connected to Qualisys")

    async def take_data(self):
        await self.setup()
        while True:
            tic = asyncio.get_event_loop().time()
            packet = await self.interface.get_current_frame(components=['3d', 'force', 'analog'])

            # data recuperation
            """
            force_plates = packet.get_force()[1]
            dict_force = {f"plate_{plate.id}": forces
                          for plate, forces in force_plates}
            """
            """
            headers, markers = packet.get_3d_markers()
            dict_marker = {
                f"markers{[i]}": [marker.x, marker.y, marker.z]
                for i, marker in enumerate(markers)}
            """

            _, analog = packet.get_analog()
            # FootSwitch branché directement sur le boitier analog sur le canal 16
            if analog[1] and self.phase_detection_method is "analog":
                foot_switch_processor.gait_phase_detection(footswitch_data=analog[1][self.analog_channel_footswitch-1][2][0])

            if analog[1] and self.phase_detection_method is "emg":
                data = analog[1]
                emg_data = []
                for device, sample, channel in data:
                    if device.id == 2:
                        channel_index = data.index((device, sample, channel)) % device.channel_count  # Numéro du canal
                        # Ajouter les données au bon channel
                        if channel_index not in emg_data:
                            emg_data[channel_index] = []
                        emg_data[channel_index].extend(channel.samples)
                foot_switch_emg_processor.heel_off_detection(emg_data[self.emg_pied_droit-1], 1)
                foot_switch_emg_processor.heel_off_detection(emg_data[self.emg_pied_gauche-1], 2)

            # combined_dict = {**dict_marker, **dict_force, **dict_footswitch}

            loop_time = time.perf_counter() - tic
            real_time_to_sleep = (1 / self.system_rate) - loop_time
            if real_time_to_sleep > 0:
                await asyncio.sleep(real_time_to_sleep)


if __name__ == "__main__":
    processor = DataReceiver()
    asyncio.run(processor.take_data())
