import numpy as np
from Stim_P24.Stim_Interface import StimInterfaceWidget

class FootSwitchDataProcessor:
    def __init__(self):
        self.previous_data = []
        self.stimulator_function = StimInterfaceWidget()

    def gait_phase_detection(self, foot_switch_data):
        if not self.previous_data:
            data = foot_switch_data
        else:
            data = np.concatenate(([self.previous_data], foot_switch_data))

        tension_change = np.diff(data)

        if np.any((-1.6 <= tension_change) & (tension_change <= -1.2)):
            print('Send stim channel 5 to 8')
            if self.stimulator_function.stimulator_is_active is True:
                self.stimulator_function.call_start_stimulation([5, 6, 7, 8])

        if np.any((-0.8 <= tension_change) & (tension_change <= -0.4)):
            if self.stimulator_function.stimulator_is_active is True:
                self.stimulator_function.call_start_stimulation([1, 2, 3, 4])
            print('Send stim channel 5 to 8')

        if np.any((1.2 <= tension_change) & (tension_change <= 1.6)):
            print('Left heel strike')

        if np.any((0.4 <= tension_change) & (tension_change <= 0.8)):
            print('Right heel strike')

        if np.any((-3 <= tension_change) & (tension_change <= -2.6)):
            if self.stimulator_function.stimulator_is_active is True:
                self.stimulator_function.call_pause_stimulation()
            print('Left toe off')

        if np.any((-0.3 <= tension_change) & (tension_change <= -0.05)):
            if self.stimulator_function.stimulator_is_active is True:
                self.stimulator_function.call_pause_stimulation()
            print('Right toe off')

        self.previous_data = data[-1]
