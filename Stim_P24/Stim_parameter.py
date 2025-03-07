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
    QLabel
)
import sys
import logging
from Stim_P24.Stim_parameter import StimulatorSetUp


# Configurer le logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class StimInterfaceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Interface Stimulation"
        self.channel_inputs = {}
        self.stimulateur_com = StimulatorSetUp()
        self.dolookneedsendstim = False
        self.init_ui()

    def init_ui(self):
        """Initialisation de l'interface utilisateur."""
        self.setWindowTitle(self.title)
        layout = QVBoxLayout(self)

        # Configuration des canaux de stimulation
        layout.addWidget(self.create_channel_config_group())

        # Contrôles de stimulation
        layout.addLayout(self.create_stimulation_controls())

        self.setLayout(layout)

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
        self.activate_button = QPushButton("Activer Stimulateur")
        self.activate_button.clicked.connect(self.stimulateur_com.activate_stimulator)
        # Use lambda to pass arguments correctly to the method
        self.update_button = QPushButton("Actualiser Paramètre Stim")
        self.update_button.clicked.connect(
            lambda: self.stimulateur_com.update_stimulation_parameter(self.channel_inputs))

        self.start_button = QPushButton("Envoyer Stimulation")
        self.start_button.clicked.connect(
            lambda: self.stimulateur_com.start_stimulation([1, 2, 3, 4, 5, 6, 7, 8]))

        self.stop_button = QPushButton("Arrêter Stimuleur")
        self.stop_button.clicked.connect(self.stimulateur_com.stop_stimulator)

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
        selected_channels = [
            i + 1 for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()
        ]

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = StimInterfaceWidget()
    widget.show()
    sys.exit(app.exec_())
