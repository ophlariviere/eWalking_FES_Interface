import numpy as np
from Stim_P24.Stim_parameter import StimulatorSetUp


class FootSwitchEMGProcessor:
    def __init__(self):
        self.stimulator_function = StimulatorSetUp()
        self.sendStim = {1: False, 2: False}

    def heel_off_detection(self, data_foot_switch_heel, data_foot_switch_toe, foot_num):
        if (np.mean(np.abs(data_foot_switch_heel)) > 1000 and np.mean(np.abs(data_foot_switch_toe)) < 500 and
                self.sendStim[foot_num] is False):
            channel_to_stim = [1, 2, 3, 4] if foot_num == 1 else [5, 6, 7, 8]
            self.stimulator_function.start_stimulation(channel_to_stim)
            self.sendStim[foot_num] = True

        if (np.mean(np.abs(data_foot_switch_heel)) > 1000 and np.mean(np.abs(data_foot_switch_toe)) < 500 and
                self.sendStim[foot_num] is False):
            self.stimulator_function.pause_stimulation()
            self.sendStim[foot_num] = False
