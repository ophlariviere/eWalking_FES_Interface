import numpy as np
from Stim_P24.Stim_parameter import StimulatorSetUp


class FootSwitchEMGProcessor:
    def __init__(self):
        self.stimulator_function = StimulatorSetUp()
        self.sendStim = {1: False, 2: False}

    def heel_off_detection(self, data_foot_switch_heel, data_foot_switch_toe, foot_num):
        print(f"{np.nanmean(np.abs(data_foot_switch_heel))} and {np.nanmean(np.abs(data_foot_switch_toe))}")
        if (np.nanmean(np.abs(data_foot_switch_heel)) > 1000 and (np.nanmean(np.abs(data_foot_switch_toe)) < 100) and
                self.sendStim[foot_num] is False):
            channel_to_stim = [1, 2, 3, 4] if foot_num == 1 else [5, 6, 7, 8]
            # self.stimulator_function.start_stimulation(channel_to_stim)
            print('Send stim')
            self.sendStim[foot_num] = True

        if (np.nanmean(np.abs(data_foot_switch_heel)) > 1000 and np.nanmean(np.abs(data_foot_switch_toe)) > 1000 and
                self.sendStim[foot_num] is True):
            # self.stimulator_function.pause_stimulation()
            print('Stop stim')
            self.sendStim[foot_num] = False
