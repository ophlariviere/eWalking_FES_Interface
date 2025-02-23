import numpy as np
from Stim_P24.Stim_parameter import StimulatorSetUp


class FootSwitchEMGProcessor:
    def __init__(self):
        self.stimulatorfonction = StimulatorSetUp()
        self.sendStim = {1: False, 2: False}

    def heel_off_detection(self, data_footswitch, piednum):
        if np.any(data_footswitch < 200) and self.sendStim[piednum] is False:
            channel_to_stim = [1, 2, 3, 4] if piednum == 1 else [5, 6, 7, 8]
            self.stimulatorfonction.start_stimulation(channel_to_stim)
            self.sendStim[piednum] = True

        if np.any(data_footswitch > 1000) and self.sendStim[piednum] is True:
            self.stimulatorfonction.pause_stimulation()
            self.sendStim[piednum] = False
