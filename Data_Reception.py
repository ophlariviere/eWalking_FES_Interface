import json
import zmq
import asyncio
import qtm_rt
import timeit
from FootSwitch.FootSwitchDataProcess import FootSwitchDataProcessor
foot_switch_processor = FootSwitchDataProcessor()
import time
import asyncio

class DataReceiver:
    def __init__(self, server_ip="192.168.0.1", port=7, data_path="example\\walkAll_LAO01_Cond10.bio",
                 model_path="example\\LAO.bioMod",
                 threshold=30, system_rate=100, device_rate=2000, nb_markers=53, nb_seconds=1):

        # Paramètres
        self.system_rate = system_rate
        self.device_rate = device_rate
        self.interface = None


    async def setup(self):
        """Setup connection to QTM"""
        self.interface = await qtm_rt.connect("192.168.254.1")
        if self.interface is None:
            return
        print("Connected to Qualisys")
        is_initialized = True
        qualisys_client = True

    async def take_data(self):
        await self.setup()
        while True:
            tic = asyncio.get_event_loop().time()
            packet = await self.interface.get_current_frame(components=['3d', 'force', 'analog'])
            # data recuperation
            headers, markers = packet.get_3d_markers()
            force_plates = packet.get_force()[1]
            analog = packet.get_analog()

            dict_force = {f"plate_{plate.id}": forces
                          for plate, forces in force_plates}
            dict_marker = {
                f"markers{[i]}": [marker.x, marker.y, marker.z]
                for i, marker in enumerate(markers)
            }

            if analog[1]:
                dict_footswitch = {
                    'footswitch_data': analog[1][15][2][0]
                }

                foot_switch_processor.gait_phase_detection(footswitch_data=analog[1][15][2][0])

            #combined_dict = {**dict_marker, **dict_force, **dict_footswitch}

            loop_time = time.perf_counter() - tic
            real_time_to_sleep = (1 / self.system_rate) - loop_time
            if real_time_to_sleep > 0:
                await asyncio.sleep(real_time_to_sleep)



if __name__ == "__main__":
    """
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()
    """
    processor = DataReceiver()
    asyncio.run(processor.take_data())
