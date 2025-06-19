import sys
import json
import csv
import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QFileDialog,
)
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer

CONFIG_FILE = "config.json"  # Fichier pour stocker les dimensions sauvegardées


class TreadmillInterface(QWidget):
    def __init__(self):
        super().__init__()

        # Charger la configuration sauvegardée (si disponible)
        self.load_config()

        # Position du COP
        self.cop_x = 0  # Centre sur l'axe X (de -0.5 à +0.5)
        self.cop_y = 0  # Centre sur l'axe Y (de 0 à 1.53)

        # Gestion de l'enregistrement
        self.is_recording = False
        self.data_log = []

        # **Création des boutons**
        self.record_button = QPushButton("Record")
        self.start_button = QPushButton("START")
        self.stop_button = QPushButton("STOP")

        # **Styles des boutons**
        self.record_button.setStyleSheet("background-color: lightgray; font-size: 14px; padding: 5px;")
        self.start_button.setStyleSheet("background-color: lightgreen; font-size: 16px; padding: 5px;")
        self.stop_button.setStyleSheet("background-color: lightcoral; font-size: 16px; padding: 5px;")

        # **Label vitesse**
        self.speed_label = QLabel("Vitesse : 0.00 m/s")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setStyleSheet(
            "font-size: 16px; background-color: lightblue; border-radius: 5px; padding: 5px;"
        )

        # Label COP x et Y
        self.cop_x_label = QLabel("COP X : 0.00 m")
        self.cop_x_label.setAlignment(Qt.AlignCenter)
        self.cop_x_label.setStyleSheet(
            "font-size: 16px; background-color: lightyellow; border-radius: 5px; padding: 5px;"
        )

        self.cop_y_label = QLabel("COP Y : 0.00 m")
        self.cop_y_label.setAlignment(Qt.AlignCenter)
        self.cop_y_label.setStyleSheet(
            "font-size: 16px; background-color: lightyellow; border-radius: 5px; padding: 5px;"
        )

        # **Disposition des widgets avec des layouts**
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.record_button)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        start_layout = QHBoxLayout()
        start_layout.addStretch()
        start_layout.addWidget(self.start_button)
        start_layout.addStretch()
        layout.addLayout(start_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addStretch()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addStretch()
        layout.addLayout(speed_layout)

        layout.addStretch()

        # **Disposition horizontale des labels COP sous le tapis**
        cop_layout = QHBoxLayout()
        cop_layout.addStretch()
        cop_layout.addWidget(self.cop_x_label)
        cop_layout.addWidget(self.cop_y_label)
        cop_layout.addStretch()
        layout.addLayout(cop_layout)

        stop_layout = QHBoxLayout()
        stop_layout.addStretch()
        stop_layout.addWidget(self.stop_button)
        stop_layout.addStretch()
        layout.addLayout(stop_layout)

        self.setLayout(layout)

        # Charger les positions des boutons si elles existent
        self.restore_positions()
        self.record_button.clicked.connect(self.toggle_recording)

        self.show()

    def paintEvent(self, event):
        """Dessine le tapis roulant, les lignes et le COP avec les zones optimales et limites."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Dimensions adaptées à la fenêtre
        width = self.width()
        height = self.height()

        treadmill_top = int(height * 0.25)
        treadmill_bottom = int(height * 0.75)
        treadmill_left = int(width * 0.2)
        treadmill_right = int(width * 0.8)
        treadmill_center_x = (treadmill_left + treadmill_right) // 2
        treadmill_height = treadmill_bottom - treadmill_top

        # **Dessiner le tapis roulant**
        painter.setBrush(QColor(200, 200, 200))
        painter.drawRect(treadmill_left, treadmill_top, treadmill_right - treadmill_left, treadmill_height)

        # **Ligne médiale horizontale à y = 0.7**
        pen_blue = QPen(QColor(100, 100, 255), 2, Qt.SolidLine)
        painter.setPen(pen_blue)
        medial_y = treadmill_top + treadmill_height - int(0.7 * treadmill_height / 1.53)
        painter.drawLine(treadmill_left, medial_y, treadmill_right, medial_y)

        # **Zone optimale en vert (y de 0.5 à 0.9)**
        painter.setBrush(QColor(100, 200, 100, 100))  # Vert transparent
        optimal_top = treadmill_top + treadmill_height - int(0.9 * treadmill_height / 1.53)
        optimal_bottom = treadmill_top + treadmill_height - int(0.5 * treadmill_height / 1.53)
        painter.drawRect(treadmill_left, optimal_top, treadmill_right - treadmill_left, optimal_bottom - optimal_top)

        # **Lignes limites avant et arrière**
        pen_red = QPen(QColor(255, 100, 100), 2, Qt.DashLine)
        painter.setPen(pen_red)

        # Limite avant (y = 1.1)
        limit_front = treadmill_top + treadmill_height - int(1.2 * treadmill_height / 1.53)
        painter.drawLine(treadmill_left, limit_front, treadmill_right, limit_front)

        # Limite arrière (y = 0.2)
        limit_back = treadmill_top + treadmill_height - int(0.2 * treadmill_height / 1.53)
        painter.drawLine(treadmill_left, limit_back, treadmill_right, limit_back)

        # **Ligne centrale verticale (x = 0)**
        pen_black = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(pen_black)
        painter.drawLine(treadmill_center_x, treadmill_top, treadmill_center_x, treadmill_bottom)

        # **Affichage du COP (inversion de y)**
        x_pos = treadmill_center_x + int(self.cop_x * (treadmill_right - treadmill_left) / 2)
        y_pos = treadmill_top + treadmill_height - int(self.cop_y * treadmill_height / 1.53)  # Inversion de Y

        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(x_pos - 5, y_pos - 5, 10, 10)

    def update_cop(self, cop_x, cop_y):
        """Met à jour la position du COP et rafraîchit l'affichage"""
        self.cop_x = max(-0.5, min(0.5, cop_x))  # Contraindre cop_x entre -0.5 et +0.5
        self.cop_y = max(0, min(1.53, cop_y))  # Contraindre cop_y entre 0 et 1.53
        self.update()  # Rafraîchit l'interface graphique

        # **Mise à jour des labels COP**
        self.cop_x_label.setText(f"COP X : {self.cop_x:.2f} m")
        self.cop_y_label.setText(f"COP Y : {self.cop_y:.2f} m")

    def log_data(self, step, treadmill_speed, treadmill_acceleration, cop_measured, cop_estimated):
        """Ajoute une ligne de données à la liste."""
        if self.is_recording:
            self.data_log.append([step, treadmill_speed, treadmill_acceleration, cop_measured, cop_estimated])

    def closeEvent(self, event):
        """Sauvegarde la configuration avant de fermer l'application."""
        self.save_config()
        event.accept()

    def load_config(self):
        """Charge la configuration si elle existe, sinon elle sera créée après le premier redimensionnement."""
        try:
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)
                self.resize(*config["window_size"])  # Appliquer la dernière taille connue
                self.button_positions = config
        except FileNotFoundError:
            self.button_positions = None  # Le fichier sera créé plus tard

    def save_config(self):
        """Sauvegarde la taille de la fenêtre et la position des boutons."""
        config = {
            "window_size": [self.width(), self.height()],  # Taille actuelle de la fenêtre
            "record_button": [self.record_button.x(), self.record_button.y()],
            "start_button": [self.start_button.x(), self.start_button.y()],
            "stop_button": [self.stop_button.x(), self.stop_button.y()],
        }

        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file)

    def restore_positions(self):
        """Restaure les positions des boutons si une configuration a été trouvée."""
        if hasattr(self, "button_positions") and self.button_positions:
            for button_name in ["record_button", "start_button", "stop_button"]:
                if (
                    button_name in self.button_positions
                    and isinstance(self.button_positions[button_name], list)
                    and len(self.button_positions[button_name]) == 2
                ):
                    getattr(self, button_name).move(*self.button_positions[button_name])

    def toggle_recording(self):
        """Active ou désactive l'enregistrement des données."""
        self.is_recording = not self.is_recording
        if not self.is_recording:
            self.auto_export_csv()  # Exporte automatiquement

        if self.is_recording:
            self.record_button.setStyleSheet("background-color: red; font-size: 14px; padding: 5px;")
            self.record_button.setText("Recording...")
            self.data_log = []  # Réinitialisation des données
        else:
            self.record_button.setStyleSheet("background-color: lightgray; font-size: 14px; padding: 5px;")
            self.record_button.setText("Record")

    def auto_export_csv(self):
        """Exporte automatiquement les données enregistrées en CSV après l'arrêt de l'enregistrement."""
        if not self.data_log:
            return  # Pas de données à sauvegarder

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
        default_filename = f"acquisition_{timestamp}.csv"

        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer CSV", default_filename, "Fichiers CSV (*.csv)")

        if file_path:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Treadmill_speed", "Treadmill_acceleration", "COP_measured", "COP_filtered"])
                writer.writerows([row[1:] for row in self.data_log])  # Exclut la première colonne (Step)

            # Confirmation visuelle avec bouton "Record" en vert 3 sec
            self.record_button.setStyleSheet("background-color: green; font-size: 14px; padding: 5px;")
            QTimer.singleShot(
                3000,
                lambda: self.record_button.setStyleSheet("background-color: lightgray; font-size: 14px; padding: 5px;"),
            )

    def resizeEvent(self, event):
        """Met à jour automatiquement la taille de la fenêtre dans config.json lorsqu'on la redimensionne."""
        self.save_config()  # Sauvegarde immédiate de la nouvelle taille
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Sauvegarde la configuration avant de fermer l'application."""
        self.save_config()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    interface = TreadmillInterface()
    sys.exit(app.exec_())
