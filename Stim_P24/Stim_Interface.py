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
    QFileDialog
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
        self.subject_mass = 700
        self.process_id = False
        self.init_ui()
        self.stimulator = None
        self.stimulator_is_active = False
        self.stimulator_is_sending_stim = False
        self.stimulator_parameters = {}
        self.foot_emg = {}
        self.num_config = 0
        self.model = None

    def init_ui(self):
        """Initialisation de l'interface utilisateur."""
        self.setWindowTitle(self.title)
        layout = QVBoxLayout(self)

        # Configuration channel emg
        # layout.addWidget(self.create_emg_num_for_foot())
        layout.addWidget(self.create_process_and_subject_info())

        # Configuration des canaux de stimulation
        layout.addWidget(self.create_channel_config_group())

        # Contrôles de stimulation
        layout.addLayout(self.create_stimulation_controls())

        self.setLayout(layout)

    def create_process_and_subject_info(self):
        # Créer un QGroupBox pour encapsuler les champs de configuration
        groupbox = QGroupBox("General information")

        # Créer un layout principal
        main_layout = QVBoxLayout()

        # Layout pour les orteils
        general_layout = QHBoxLayout()  # Layout horizontal pour les orteils
        subject_mass = QSpinBox()
        subject_mass.setRange(0, 200)
        subject_mass.setPrefix("Subject mass: ")

        model_selection = QPushButton("Choisir un fichier", self)
        model_selection.clicked.connect(self.open_filename_dialog)

        self.process_dyn=QCheckBox('Process IK and ID')
        self.process_dyn.setChecked(False)
        self.process_dyn.stateChanged.connect(lambda state: self.info_dyn)


        general_layout.addWidget(subject_mass)
        general_layout.addWidget(model_selection)
        general_layout.addWidget(self.process_dyn)

        # Créer un bouton OK et connecter la fonction d'enregistrement
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(
            lambda: self.save_subject_info(subject_mass.value())
        )
        # Ajouter le bouton au layout
        main_layout.addWidget(ok_button)

        # Ajouter les sous layouts (orteils et talons) au layout principal
        main_layout.addLayout(general_layout)

        # Assigner le layout au QGroupBox
        groupbox.setLayout(main_layout)

        # Retourner le QGroupBox complet
        return groupbox

    def save_subject_info(self,mass_value):
        # Enregistrer les valeurs dans foot_emg
        self.subject_mass = mass_value * 9.81

    def open_filename_dialog(self):
        # Ouvre la boîte de dialogue pour sélectionner un fichier
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier", "",
                                                   "Tous les fichiers (*);;Fichiers texte (*.txt)", options=options)
        if file_name:
            # Affiche le nom du fichier dans l'étiquette
            self.label.setText(f"Fichier sélectionné : {file_name}")
            self.model = biorbd.Model(file_name)


    def info_dyn(self):
        if self.process_id.isChecked():
            if self.model:
                self.process_id = True
        else:
            print('You need to load a bioMod to process IK, ID')
            self.process_dyn.setChecked(False)
            self.process_id = False




    """Visu Stim"""

    def create_channel_config_group(self):
        """Créer un groupbox pour configurer les canaux."""
        groupbox = QGroupBox("Configurer les canaux")
        layout = QVBoxLayout()  # Ce layout contiendra les widgets pour les canaux

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
        activate_button = QPushButton("Activer Stimulateur")
        activate_button.clicked.connect(self.activate_stimulator)
        # Use lambda to pass arguments correctly to the method
        update_button = QPushButton("Actualiser Paramètre Stim")
        update_button.clicked.connect(self.update_stimulation_parameter)

        start_button = QPushButton("Envoyer Stimulation")
        start_button.clicked.connect(lambda: self.call_start_stimulation([1, 2, 3, 4, 5, 6, 7, 8]))

        stop_button = QPushButton("Arrêter Stimulateur")
        stop_button.clicked.connect(self.stop_stimulator)

        self.check_pause_stim = QCheckBox("Stop tying send stim")
        self.check_pause_stim.setChecked(True)
        self.check_pause_stim.stateChanged.connect(self.pause_fonction_to_send_stim)
        layout.addWidget(self.check_pause_stim)
        layout.addWidget(activate_button)
        layout.addWidget(start_button)
        layout.addWidget(update_button)
        layout.addWidget(stop_button)
        return layout

    def pause_fonction_to_send_stim(self):
        self.dolookneedsendstim = not self.check_pause_stim.isChecked()

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


