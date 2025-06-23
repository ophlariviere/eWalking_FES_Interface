import threading
import asyncio
import time
from Bertec_self_paced import treadmill_remote
from Com import QualysisReception_SendAll_ToServer as qualisys_receiver


def run_treadmill():
    """Démarre le contrôle du tapis roulant"""
    app = treadmill_remote.interface.QApplication([])
    estimator = treadmill_remote.StateEstimator()
    controller = treadmill_remote.LQGController()
    gui = treadmill_remote.TreadmillAIInterface(estimator, controller)
    gui.show()
    app.exec_()
    # treadmill_remote.app.exec_()  # Démarre l'interface graphique du tapis


def run_qualisys():
    """Démarre l'écoute des données Qualisys en mode asynchrone"""
    processor = qualisys_receiver.QualisysDataReceiver()
    asyncio.run(processor.listen_for_data())  # Exécute la boucle asynchrone


if __name__ == "__main__":
    # Créer les threads
    treadmill_thread = threading.Thread(target=run_treadmill, daemon=True)
    qualisys_thread = threading.Thread(target=run_qualisys, daemon=True)

    # Démarrer les threads
    treadmill_thread.start()
    qualisys_thread.start()

    # Maintenir le script actif tant que les threads tournent
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur.")
