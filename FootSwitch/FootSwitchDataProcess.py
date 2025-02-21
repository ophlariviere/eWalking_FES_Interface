import numpy as np
from Stim_P24.Stim_parameter import StimulatorSetUp


class FootSwitchDataProcessor:
    def __init__(self):
        self.previousdata = []
        self.stimulatorfonction = StimulatorSetUp()

    def gait_phase_detection(self, footswitch_data):
        if not self.previousdata:
            data = footswitch_data
        else:
            data = np.concatenate(([self.previousdata], footswitch_data))

        tension_change = np.diff(data)

        if np.any((-1.6 <= tension_change) & (tension_change <= -1.2)):
            print('Send stim channel 5 to 8')
            if self.stimulatorfonction.stimulator_is_active is True:
                self.stimulatorfonction.start_stimulation(channel_to_send=[5, 6, 7, 8])

        if np.any((-0.8 <= tension_change) & (tension_change <= -0.4)):
            if self.stimulatorfonction.stimulator_is_active is True:
                self.stimulatorfonction.start_stimulation(channel_to_send=[1, 2, 3, 4])
            print('Send stim channel 5 to 8')

        if np.any((1.2 <= tension_change) & (tension_change <= 1.6)):
            print('Gauche heel strike')

        if np.any((0.4 <= tension_change) & (tension_change <= 0.8)):
            print('Droite heel strike')

        if np.any((-3 <= tension_change) & (tension_change <= -2.6)):
            if self.stimulatorfonction.stimulator_is_active is True:
                self.stimulatorfonction.pause_stimulation()
            print('Gauche toe off')

        if np.any((-0.3 <= tension_change) & (tension_change <= -0.05)):
            if self.stimulatorfonction.stimulator_is_active is True:
                self.stimulatorfonction.pause_stimulation()
            print('Droite toe off')

        self.previousdata = data[-1]
