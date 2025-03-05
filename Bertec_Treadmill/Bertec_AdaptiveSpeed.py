import BertecRemoteControl
import time



def estimate_speed():
    speed = 1.5
    return speed


remote = BertecRemoteControl.RemoteControl()
res = remote.start_connection()

print(res)
while True:
    print("Get_force")
    res = remote.get_force_data()
    speed = estimate_speed()
    vel = str(speed).replace(".", ",")

    params = {
        'leftVel': vel,
        'leftAccel': '0,1',
        'leftDecel': '0,1',
        'rightVel': vel,
        'rightAccel': '0,1',
        'rightDecel': '0,1',
        }
    res = remote.run_treadmill(params['leftVel'], params['leftAccel'], params['leftDecel'], params['rightVel'],
                               params['rightAccel'], params['rightDecel'])

    time.sleep(0.01)


remote.stop_connection()
