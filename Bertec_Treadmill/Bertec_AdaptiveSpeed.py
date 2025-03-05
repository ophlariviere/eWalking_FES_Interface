import BertecRemoteControl

remote = BertecRemoteControl.RemoteControl()
res = remote.start_connection()
print(res)
while True:
    print("Get_force")
    res = remote.get_force_data()


remote.stop_connection()