from pysciencemode import Device, Modes, Channel
from pysciencemode import RehastimP24 as St
import logging
from Stim_Interface import StimInterfaceWidget


class StimulatorSetUp:
    def __init__(self):
        super().__init__()
        self.stimulator = None
        self.stimulator_is_active = False
        self.stimulator_is_sending_stim = False
        self.stimulator_parameters = {}
        self.num_config = 0
        self.interface = StimInterfaceWidget

    def activate_stimulateur(self):
        if self.stimulator is None:
            self.stimulator = St(port="COM3", show_log="Status")
            self.stimulator_is_active = False

    def start_stimulation(self, channel_to_send):
        try:
            if self.stimulator is None:
                logging.warning(
                    "Stimulateur non initialisé. Veuillez le configurer avant de démarrer."
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
                logging.info(f"Stimulation démarrée sur les canaux {channel_to_send}")

        except Exception as e:
            logging.error(f"Erreur lors du démarrage de la stimulation : {e}")

    def pause_stimulation(self):
        try:
            if self.stimulator:
                self.stimulator.end_stimulation()
                self.stimulator_is_sending_stim = False
                logging.info("Stimulation arrêtée.")
            else:
                logging.warning("Aucun stimulateur actif à arrêter.")
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt de la stimulation : {e}")

    def stop_stimulateur(self):
        try:
            if self.stimulator:
                self.pause_stimulation()
                self.stimulator_is_sending_stim = False
                self.stimulator.close_port()
                self.stimulator_is_active = False
                self.stimulator = None
                logging.info("Stimulateur arrêtée.")
            else:
                logging.warning("Aucun stimulateur actif à arrêter.")
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt de la stimulation : {e}")

    def update_stimulation_parameter(self):
        """Met à jour la stimulation."""
        self.num_config += 1

        if self.stimulator is not None:
            for channel, inputs in self.interface.channel_inputs.items():
                # Vérifiez si le canal existe dans stimconfig, sinon, initialisez-le
                if channel not in self.stimulator_parameters:
                    self.stimulator_parameters[channel] = {
                        "name": "",
                        "amplitude": 0,
                        "pulse_width": 0,
                        "frequency": 0,
                        "mode": "",
                        "device_type": None,
                    }

                # Mettez à jour les valeurs de configuration
                self.stimulator_parameters[channel]["name"] = inputs["name_input"].text()
                self.stimulator_parameters[channel]["amplitude"] = inputs["amplitude_input"].value()
                self.stimulator_parameters[channel]["pulse_width"] = inputs["pulse_width_input"].value()
                self.stimulator_parameters[channel]["frequency"] = inputs["frequency_input"].value()
                self.stimulator_parameters[channel]["mode"] = inputs["mode_input"].currentText()
                self.stimulator_parameters[channel]["device_type"] = Device.Rehastimp24
