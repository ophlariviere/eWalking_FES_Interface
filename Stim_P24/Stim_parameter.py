from pysciencemode import Device, Modes, Channel
from pysciencemode import RehastimP24 as St
import logging


class StimulatorSetUp:
    def __init__(self):
        super().__init__()
        from Stim_P24.Stim_Interface import StimInterfaceWidget
        self.stimulator = None
        self.stimulator_is_active = False
        self.stimulator_is_sending_stim = False
        self.stimulator_parameters = {}
        self.num_config = 0
        self.interface = StimInterfaceWidget

    def activate_stimulator(self):
        if self.stimulator is None:
            self.stimulator = St(port="COM3", show_log="Status")
            self.stimulator_is_active = True

    def start_stimulation(self, channel_to_send):
        try:
            if self.stimulator is None:
                logging.warning(
                    "Stimulator non initialised. Please initialised stimulator before sending stim."
                )
                return
            channels_instructions = []
            for channel, inputs in self.interface.channel_inputs.items():
                if channel in channel_to_send:
                    channel = Channel(
                        no_channel=channel,
                        name=self.stimulator_parameters[channel]["name"].text(),
                        amplitude=self.stimulator_parameters[channel]["amplitude"].value(),
                        pulse_width=self.stimulator_parameters[channel]["pulse_width"].value(),
                        frequency=self.stimulator_parameters[channel]["frequency"].value(),
                        mode=Modes.SINGLE,
                        device_type=Device.Rehastimp24,
                    )
                    channels_instructions.append(channel)
            if self.stimulator_is_sending_stim is True:
                self.pause_stimulation()
            if channels_instructions:
                self.stimulator.init_stimulation(list_channels=channels_instructions)
                self.stimulator.update_stimulation(upd_list_channels=channels_instructions)
                self.stimulator.start_stimulation(upd_list_channels=channels_instructions)
                self.stimulator_is_sending_stim = True
                logging.info(f"Stimulation start on channel {channel_to_send}")

        except Exception as e:
            logging.error(f"Error when sending stimulation : {e}")

    def pause_stimulation(self):
        try:
            if self.stimulator:
                self.stimulator.end_stimulation()
                self.stimulator_is_sending_stim = False
                logging.info("Stimulation stopped.")
            else:
                logging.warning("No stimulator is active so stimulation can't be stopped.")
        except Exception as e:
            logging.error(f"Error during stopping stimulation : {e}")

    def stop_stimulator(self):
        try:
            if self.stimulator:
                self.pause_stimulation()
                self.stimulator_is_sending_stim = False
                self.stimulator.close_port()
                self.stimulator_is_active = False
                self.stimulator = None
                logging.info("Stimulator stopped.")
            else:
                logging.warning("None stimulator to stopped.")
        except Exception as e:
            logging.error(f"Error during stopping the stimulator : {e}")

    def update_stimulation_parameter(self):
        """Met à jour la stimulation."""
        self.num_config += 1

        if self.stimulator is not None:
            for channel, inputs in self.interface.channel_inputs.items():
                # Check if channel exist yet, if not initialised it
                if channel not in self.stimulator_parameters:
                    self.stimulator_parameters[channel] = {
                        "name": "",
                        "amplitude": 0,
                        "pulse_width": 0,
                        "frequency": 0,
                        "mode": "",
                        "device_type": None,
                    }

                # Upgrade stimulation parameters value
                self.stimulator_parameters[channel]["name"] = inputs["name_input"].text()
                self.stimulator_parameters[channel]["amplitude"] = inputs["amplitude_input"].value()
                self.stimulator_parameters[channel]["pulse_width"] = inputs["pulse_width_input"].value()
                self.stimulator_parameters[channel]["frequency"] = inputs["frequency_input"].value()
                self.stimulator_parameters[channel]["mode"] = inputs["mode_input"].currentText()
                self.stimulator_parameters[channel]["device_type"] = Device.Rehastimp24
