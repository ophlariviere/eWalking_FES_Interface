import asyncio
import qtm_rt

async def on_packet(packet):
    """Callback function called for each received packet."""
    analog_data = packet.get_analog()
    if analog_data:
        # Traitement des données analogiques
        device_id, channels = analog_data
        for channel in channels:
            print(f"Channel {channel.id}: {channel.samples}")

async def setup():
    """Setup connection to QTM and start streaming."""
    connection = await qtm_rt.connect("192.168.254.1")  # Remplacez par l'adresse IP de votre système QTM
    if connection is None:
        print("Échec de la connexion à QTM.")
        return

    await connection.stream_frames(components=["analog"], on_packet=on_packet)

if __name__ == "__main__":
    asyncio.run(setup())