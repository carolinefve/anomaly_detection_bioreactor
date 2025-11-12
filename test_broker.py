import paho.mqtt.client as mqtt
import json
import os
from dotenv import load_dotenv

load_dotenv()

BROKER_ADDRESS = os.getenv("BROKER_ADDRESS") 
BROKER_PORT = int(os.getenv("BROKER_PORT"))
TOPIC = "bioreactor_sim/single_fault/telemetry/summary"


# Runs when the client successfully connects
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print(f"Connected successfully (V2 API) to broker at {BROKER_ADDRESS}")
        # Subscribe to the topic once connected
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

# Runs every time a new message is received
def on_message(client, userdata, msg):

    try:
        # Decode the message from bytes to a string, then parse as JSON
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
        
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Raw payload: {msg.payload}")


# Create a unique client ID
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "group_5_2025_client") 

client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
except Exception as e:
    print(f"Could not connect to broker: {e}")
    exit()

print("Starting network loop...")
client.loop_forever()