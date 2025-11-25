import paho.mqtt.client as mqtt
import json
import numpy as np
import os
import math
import sys
from dotenv import load_dotenv

load_dotenv()

# Connection Details 
BROKER_ADDRESS = os.getenv("BROKER_ADDRESS") 
BROKER_PORT = int(os.getenv("BROKER_PORT"))

# Topic Constants 
TOPIC_TRAINING = "bioreactor_sim/nofaults/telemetry/summary"
TOPIC_TEST_SINGLE = "bioreactor_sim/single_fault/telemetry/summary"
TOPIC_TEST_THREE = "bioreactor_sim/three_faults/telemetry/summary"

# Training Configuration 
TRAINING_SAMPLES = 300
BASELINE_FILE = "baseline_static.json"
baseline_is_trained = False

# Data Storage 
training_data = {
    "temperature": [],
    "ph": [],
    "rpm": []
}
baseline = {}

# Detection Configuration 
TAU_HIGH = 3.0
TAU_LOW = 2.8
alarm_is_active = False

scores = {
    "tp": 0, "fp": 0, "tn": 0, "fn": 0
}


current_topic = ""
current_mode = ""
target_topic_after_training = None 

# Baseline Functions 
def save_baseline(baseline_data, filename):
    try:
        with open(filename, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"\nSuccessfully saved baseline to {filename}\n")
    except Exception as e:
        print(f"Error saving baseline: {e}")

def load_baseline(filename):
    global baseline, baseline_is_trained
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                baseline = json.load(f)
            baseline_is_trained = True
            print(f"Successfully loaded baseline from {filename}")
            print(json.dumps(baseline, indent=2))
            return True
        except Exception as e:
            print(f"Error loading baseline file: {e}")
            return False
    baseline_is_trained = False
    return False

# Results Function
def save_results(scores_data, topic_string):
    try:
        test_name = topic_string.split('/')[1] 
        filename = f"results_{test_name}.json"
        
        tp = scores_data.get("tp", 0)
        fp = scores_data.get("fp", 0)
        tn = scores_data.get("tn", 0)
        fn = scores_data.get("fn", 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            "test_stream": test_name,
            "counts": scores_data,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        }
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nSuccessfully saved final results to {filename}")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error saving results: {e}")


def on_connect(client, userdata, flags, rc, properties):
    global current_topic
    if rc == 0:
        print(f"Connected successfully to broker at {BROKER_ADDRESS}")
        client.subscribe(current_topic)
        print(f"Subscribed to topic: {current_topic}")
        
        if baseline_is_trained:
            print(f"Baseline is LOADED. Starting {current_mode} detection")
        else:
            print(f"No baseline file found. Starting {current_mode}...")
            print(f"Will collect {TRAINING_SAMPLES} samples...")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global baseline_is_trained, alarm_is_active, baseline
    global current_topic, current_mode, target_topic_after_training
    
    try:
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
        
        # Training logic
        if not baseline_is_trained:
            temp_val = data.get("temperature_C", {}).get("mean")
            ph_val = data.get("pH", {}).get("mean")
            rpm_val = data.get("rpm", {}).get("mean")
            
            if temp_val is None or ph_val is None or rpm_val is None:
                print("Skipping message, missing sensor data.")
                return

            training_data["temperature"].append(temp_val)
            training_data["ph"].append(ph_val)
            training_data["rpm"].append(rpm_val)
            
            print(f"Collecting training data... {len(training_data['temperature'])} / {TRAINING_SAMPLES}", end="\n")
            

            if len(training_data["temperature"]) >= TRAINING_SAMPLES:
                print("\nTRAINING COMPLETE")
                
                for sensor_name in training_data:
                    data_list = training_data[sensor_name]
                    baseline[sensor_name] = {
                        "mean": np.mean(data_list),
                        "std": np.std(data_list)
                    }
                    print(f"Baseline for '{sensor_name}':")
                    print(f"  Mean: {baseline[sensor_name]['mean']:.4f}")
                    print(f"  Std Dev: {baseline[sensor_name]['std']:.4f}")
                
                save_baseline(baseline, BASELINE_FILE)
                
                baseline_is_trained = True
                old_topic = current_topic
                
                current_topic = target_topic_after_training
                current_mode = sys.argv[1].lower() 
                
                client.unsubscribe(old_topic)
                client.subscribe(current_topic)
                
                print(f"Baseline is LOADED. Starting {current_mode} detection")
                
        # Detection logic
        else:
            temp_val = data.get("temperature_C", {}).get("mean")
            ph_val = data.get("pH", {}).get("mean")
            rpm_val = data.get("rpm", {}).get("mean")
            
            if temp_val is None or ph_val is None or rpm_val is None:
                print("Skipping message, missing data.")
                return

            temp_z = (temp_val - baseline["temperature"]["mean"]) / baseline["temperature"]["std"]
            ph_z = (ph_val - baseline["ph"]["mean"]) / baseline["ph"]["std"]
            rpm_z = (rpm_val - baseline["rpm"]["mean"]) / baseline["rpm"]["std"]
            
            pooled_score = math.sqrt(temp_z**2 + ph_z**2 + rpm_z**2)
            
            if pooled_score > TAU_HIGH:
                alarm_is_active = True
            elif pooled_score < TAU_LOW:
                alarm_is_active = False
            
            faults_list = data.get("faults", {}).get("last_active", [])
            is_fault_present = len(faults_list) > 0

            if alarm_is_active and is_fault_present:
                scores["tp"] += 1
            elif alarm_is_active and not is_fault_present:
                scores["fp"] += 1
            elif not alarm_is_active and not is_fault_present:
                scores["tn"] += 1
            elif not alarm_is_active and is_fault_present:
                scores["fn"] += 1

            alarm_status = "!! ALARM !!" if alarm_is_active else "Normal"
            truth_status = "FAULT" if is_fault_present else "Normal"
            
            print(f"Score: {pooled_score:.2f} | T_Z: {temp_z:.1f} pH_Z: {ph_z:.1f} RPM_Z: {rpm_z:.1f} | Status: {alarm_status} | Truth: {truth_status}", end="\n")

    except Exception as e:
        print(f"Error processing message: {e}")

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        sys.exit(1)
        
    current_mode = sys.argv[1].lower()
    
    if current_mode == "single":
        target_topic_after_training = TOPIC_TEST_SINGLE
    elif current_mode == "three":
        target_topic_after_training = TOPIC_TEST_THREE
    else:
        print(f"Error: Unknown test mode '{current_mode}'.")
        sys.exit(1)

    if not load_baseline(BASELINE_FILE):
        print(f"Warning: {BASELINE_FILE} not found. Running in TRAINING mode first.")
        current_mode = "train"
        current_topic = TOPIC_TRAINING
    else:
        current_topic = target_topic_after_training
                
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"group_5_2025_{current_mode}")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
        sys.exit(1)

    print(f"Starting network loop for mode: {current_mode}")
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        if baseline_is_trained and current_mode != "train":
            save_results(scores, current_topic)
        
        client.disconnect()
        print("Disconnected from broker. Exiting.")
        sys.exit(0)