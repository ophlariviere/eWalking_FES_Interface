from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QWidget,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QLabel,
    QFileDialog,
)
import sys
import logging
import biorbd
from pysciencemode import Device, Modes, Channel
from pysciencemode import RehastimP24 as St


# Configurer le logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class StimInterfaceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Interface Stimulation"
        self.channel_inputs = {}
        self.dolookneedsendstim = False
        self.init_ui()
        self.stimulator = None
        self.stimulator_is_active = False
        self.stimulator_is_sending_stim = False
        self.stimulator_parameters = {}
        self.foot_emg = {}
        self.num_config = 0
        self.model = None
        self.mass = 70
        self.process_idik = False

    def init_ui(self):
        """Initialisation de l'interface utilisateur."""
        self.setWindowTitle(self.title)
        layout = QVBoxLayout(self)

        # Configuration channel emg
        # layout.addWidget(self.create_emg_num_for_foot())

        layout.addWidget(self.create_participant_info())

        # Configuration des canaux de stimulation
        layout.addWidget(self.create_channel_config_group())

        # Contrôles de stimulation
        layout.addLayout(self.create_stimulation_controls())

        self.setLayout(layout)

    def create_participant_info(self):
        groupbox = QGroupBox("Participant_info")
        main_sett_layout = QVBoxLayout()
        mass_layout = QHBoxLayout()
        mass = QSpinBox()
        mass.setRange(0, 400)
        mass.setPrefix("Participant mass [kg]:  ")
        ok_mass = QPushButton("OK")
        ok_mass.clicked.connect(
            lambda: self.update_mass(mass.value)
        )
        mass_layout.addWidget(mass)
        mass_layout.addWidget(ok_mass)

        model_layout = QHBoxLayout()
        self.model_label = QLabel("No file selected.")
        model_button = QPushButton("Upload File")
        model_button.clicked.connect(self.upload_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(model_button)
        model_layout.addWidget(model_button)

        self.checkbox_pro_idik = QCheckBox("Process IK/ID")
        self.checkbox_pro_idik.setChecked(False)
        self.checkbox_pro_idik.stateChanged.connect(self.need_process_idik)
        self.checkbox_pro_idik.setEnabled(False)
        model_layout.addWidget(self.checkbox_pro_idik)


        main_sett_layout.addLayout(mass_layout)
        main_sett_layout.addLayout(model_layout)
        groupbox.setLayout(main_sett_layout)
        return groupbox

    def need_process_idik(self):
        self.process_idik = True
        self.checkbox_pro_idik.setChecked(True)


    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a file")
        if file_name:
            self.model_label.setText(f"Selected File:\n{file_name}")
            self.model = biorbd.Model(file_name)
            self.checkbox_pro_idik.setEnabled(True)
            # You can now process the file as needed
            print(f"File selected: {file_name}")

    def update_mass(self, mass_value):
        self.mass = mass_value

    def create_emg_num_for_foot(self):
        # Créer un QGroupBox pour encapsuler les champs de configuration
        groupbox = QGroupBox("Configurer les canaux")

        # Créer un layout principal pour les pieds
        main_layout = QVBoxLayout()

        # Layout pour les orteils
        foot_toe_layout = QHBoxLayout()  # Layout horizontal pour les orteils
        left_toe = QSpinBox()
        left_toe.setRange(0, 16)
        left_toe.setPrefix("Left Toe: ")

        right_toe = QSpinBox()
        right_toe.setRange(0, 16)
        right_toe.setPrefix("Right Toe: ")

        foot_toe_layout.addWidget(left_toe)
        foot_toe_layout.addWidget(right_toe)

        # Layout pour les talons
        foot_heel_layout = QHBoxLayout()  # Layout horizontal pour les talons
        left_heel = QSpinBox()
        left_heel.setRange(0, 16)
        left_heel.setPrefix("Left Heel: ")

        right_heel = QSpinBox()
        right_heel.setRange(0, 16)
        right_heel.setPrefix("Right Heel: ")

        foot_heel_layout.addWidget(left_heel)
        foot_heel_layout.addWidget(right_heel)

        # Créer un bouton OK et connecter la fonction d'enregistrement
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(
            lambda: self.save_emg_values(left_toe.value(), right_toe.value(), left_heel.value(), right_heel.value())
        )

        # Ajouter le bouton au layout
        main_layout.addWidget(ok_button)

        # Ajouter les sous-layouts (orteils et talons) au layout principal
        main_layout.addLayout(foot_toe_layout)
        main_layout.addLayout(foot_heel_layout)

        # Assigner le layout au QGroupBox
        groupbox.setLayout(main_layout)

        # Retourner le QGroupBox complet
        return groupbox

    def save_emg_values(self, left_toe_value, right_toe_value, left_heel_value, right_heel_value):
        # Enregistrer les valeurs dans foot_emg
        self.foot_emg = {
            "Left Toe": left_toe_value,
            "Right Toe": right_toe_value,
            "Left Heel": left_heel_value,
            "Right Heel": right_heel_value,
        }

    """Visu Stim"""

    def create_channel_config_group(self):
        """Créer un groupbox pour configurer les canaux."""
        groupbox = QGroupBox("Configurer les canaux")
        layout = QVBoxLayout()  # Ce layout contiendra les widgets pour les canaux

        # Checkbox pour appliquer à tous les canaux
        self.copy_to_all_checkbox = QCheckBox("Appliquer à tous les canaux")
        self.copy_to_all_checkbox.setToolTip("Utilise les réglages du premier canal sélectionné pour tous les autres")
        self.copy_to_all_checkbox.stateChanged.connect(self.apply_same_settings_to_all_channels)
        layout.addWidget(self.copy_to_all_checkbox)

        # Ajouter les cases à cocher pour sélectionner les canaux
        self.checkboxes = []
        checkbox_layout = QHBoxLayout()
        for i in range(1, 9):
            checkbox = QCheckBox(f"Canal {i}")
            checkbox.stateChanged.connect(self.update_channel_inputs)
            checkbox_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        layout.addLayout(checkbox_layout)

        # Ajouter un layout vertical pour les entrées dynamiques des canaux
        self.channel_config_layout = QVBoxLayout()
        layout.addLayout(self.channel_config_layout)

        groupbox.setLayout(layout)
        return groupbox

    def create_stimulation_controls(self):
        """Créer les boutons pour contrôler la stimulation."""
        layout = QHBoxLayout()
        self.activate_button = QPushButton("Activer Stimulateur")
        self.activate_button.clicked.connect(self.activate_stimulator)
        # Use lambda to pass arguments correctly to the method
        self.update_button = QPushButton("Actualiser Paramètre Stim")
        self.update_button.clicked.connect(self.update_stimulation_parameter)

        self.start_button = QPushButton("Envoyer Stimulation")
        self.start_button.clicked.connect(lambda: self.call_start_stimulation([1, 2, 3, 4, 5, 6, 7, 8]))

        self.stop_button = QPushButton("Arrêter Stimuleur")
        self.stop_button.clicked.connect(self.stop_stimulator)

        self.checkpauseStim = QCheckBox("Stop tying send stim")
        self.checkpauseStim.setChecked(True)
        self.checkpauseStim.stateChanged.connect(self.pausefonctiontosendstim)
        layout.addWidget(self.checkpauseStim)
        layout.addWidget(self.activate_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.update_button)
        layout.addWidget(self.stop_button)
        return layout

    def pausefonctiontosendstim(self):
        # Met à jour self.dolookstimsend selon l'état de la checkbox
        self.dolookneedsendstim = not self.checkpauseStim.isChecked()

    def update_channel_inputs(self):
        """Met à jour les entrées des canaux sélectionnés sous les cases à cocher."""
        selected_channels = [i + 1 for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

        # Ajouter les nouveaux canaux sélectionnés
        for channel in selected_channels:
            if channel not in self.channel_inputs:
                # Layout pour les entrées du canal
                channel_layout = QHBoxLayout()

                # Création des widgets d'entrée pour le canal
                name_input = QLineEdit()
                name_input.setPlaceholderText(f"Canal {channel} - Nom")
                amplitude_input = QSpinBox()
                amplitude_input.setRange(0, 100)
                amplitude_input.setSuffix(" mA")
                pulse_width_input = QSpinBox()
                pulse_width_input.setRange(0, 1000)
                pulse_width_input.setSuffix(" µs")
                frequency_input = QSpinBox()
                frequency_input.setRange(0, 200)
                frequency_input.setSuffix(" Hz")
                mode_input = QComboBox()
                mode_input.addItems(["SINGLE", "DOUBLET", "TRIPLET"])

                # Ajouter les widgets au layout
                channel_layout.addWidget(QLabel(f"Canal {channel}:"))
                channel_layout.addWidget(name_input)
                channel_layout.addWidget(amplitude_input)
                channel_layout.addWidget(pulse_width_input)
                channel_layout.addWidget(frequency_input)
                channel_layout.addWidget(mode_input)

                # Ajouter le layout dans l'affichage des paramètres des canaux
                self.channel_config_layout.addLayout(channel_layout)

                # Enregistrer les widgets pour le canal sélectionné
                self.channel_inputs[channel] = {
                    "layout": channel_layout,
                    "name_input": name_input,
                    "amplitude_input": amplitude_input,
                    "pulse_width_input": pulse_width_input,
                    "frequency_input": frequency_input,
                    "mode_input": mode_input,
                }

        # Supprimer les canaux désélectionnés
        for channel in list(self.channel_inputs.keys()):
            if channel not in selected_channels:
                inputs = self.channel_inputs.pop(channel)
                layout = inputs["layout"]
                # Supprimer les widgets du layout
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                # Supprimer le layout lui-même
                self.channel_config_layout.removeItem(layout)

    def activate_stimulator(self):
        if self.stimulator is None:
            self.stimulator = St(port="COM3", show_log="Status")
            self.stimulator_is_active = True
            self.num_config = 0

    def call_start_stimulation(self, channel_to_send):
        try:
            if self.stimulator is None:
                logging.warning("Stimulator non initialised. Please initialised stimulator before sending stim.")
                return
            if self.stimulator_is_sending_stim is True:
                self.call_pause_stimulation()
            channels_instructions = [
                Channel(
                    no_channel=channel,
                    name=params["name"],
                    amplitude=params["amplitude"] if channel in channel_to_send else 0,
                    pulse_width=params["pulse_width"],
                    frequency=params["frequency"],
                    mode=Modes.SINGLE,
                    device_type=Device.Rehastimp24,
                )
                for channel, params in self.stimulator_parameters.items()
            ]

            if channels_instructions:
                self.stimulator.init_stimulation(list_channels=channels_instructions)
                self.stimulator.update_stimulation(upd_list_channels=channels_instructions)
                self.stimulator.start_stimulation(upd_list_channels=channels_instructions)
                self.stimulator_is_sending_stim = True
                logging.info(f"Stimulation start on channel {channel_to_send}")

        except Exception as e:
            logging.error(f"Error when sending stimulation : {e}")

    def call_pause_stimulation(self):
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
                self.call_pause_stimulation()
                self.stimulator_is_sending_stim = False
                self.stimulator.close_port()
                self.stimulator_is_active = False
                self.stimulator = None
                logging.info("Stimulator stopped.")
            else:
                logging.warning("None stimulator to stopped.")
        except Exception as e:
            logging.error(f"Error during stopping the stimulator : {e}")

    def apply_same_settings_to_all_channels(self):
        if not self.copy_to_all_checkbox.isChecked():
            return

        selected_channels = list(self.channel_inputs.keys())
        if len(selected_channels) < 2:
            return  # rien à copier s’il n’y a qu’un seul canal sélectionné

        ref = self.channel_inputs[selected_channels[0]]  # on prend le premier canal comme référence
        for ch in selected_channels[1:]:
            self.channel_inputs[ch]["name_input"].setText(ref["name_input"].text())
            self.channel_inputs[ch]["amplitude_input"].setValue(ref["amplitude_input"].value())
            self.channel_inputs[ch]["pulse_width_input"].setValue(ref["pulse_width_input"].value())
            self.channel_inputs[ch]["frequency_input"].setValue(ref["frequency_input"].value())
            self.channel_inputs[ch]["mode_input"].setCurrentIndex(ref["mode_input"].currentIndex())

    def update_stimulation_parameter(self):
        """Met à jour la stimulation."""
        self.num_config += 1
        self.stimulator_parameters = {}
        if self.stimulator is not None:
            for channel, inputs in self.channel_inputs.items():
                # Check if channel exist yet, if not initialised it
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
            print("Stimulator parameter updated")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = StimInterfaceWidget()
    widget.show()
    sys.exit(app.exec_())
