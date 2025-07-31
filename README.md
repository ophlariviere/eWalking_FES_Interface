# eWalking_FES_Interface

This repo is used to handle functional electric stimulation (FES) for the eWalking project. It is used to control the 
FES system (P24) automatically and to visualize the motion capture data in real time.
This repo contains codes for different data collection types:

1. Automatic stimulation during the propulsion phase (Thomas' PhD project):
- Start the motion capture trial (data collection computer)
- Allow the Bertec treadmill to be controlled remotely (data collection computer)
- Run Main_Bertec_Cometa.py (data collection computer)
- Run Com/main.py (stimulation computer)

2. Bayesian optimisation of the stimulation parameters (Eve+Ophélie postdoc project):
- Start the Redis database from Docker (stimulation computer)
- Start the motion capture trial (data collection computer)
- Run Com/QualisysReception_eve (data collection computer)
- Run Com/main_Redis.py (stimulation computer)


## How to install
```
conda install -c conda-forge pyqt biorbd biosiglive pysciencemode scikit-optimize scipy ezc3d pyzmq
pip install qtm-rt
```

## Redis setup
1. Install Docker (https://www.docker.com/get-started/)
2. Install redis (https://redis.io/docs/latest/operate/rs/installing-upgrading/quickstarts/docker-quickstart/)



