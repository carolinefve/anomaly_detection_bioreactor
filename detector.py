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

TOPIC_TRAINING = "bioreactor_sim/nofaults/telemetry/summary"
TOPIC_TEST_SINGLE = "bioreactor_sim/single_fault/telemetry/summary"
TOPIC_TEST_THREE = "bioreactor_sim/three_faults/telemetry/summary"
TOPIC_TEST_VARIABLE = "bioreactor_sim/variable_setpoints/telemetry/summary"

# Training Configuration
TRAINING_SAMPLES = 300
BASELINE_FILE = "baseline.json"
baseline_is_trained = False

# Data Storage 
training_data = {
    "temperature": [],
    "ph": [],
    "rpm": []
}
baseline = {}

# Detection & Scoring Configuration
TAU_HIGH = 3.0
TAU_LOW = 2.8
alarm_is_active = False

scores = {
    "tp": 0, "fp": 0, "tn": 0, "fn": 0
}

# Store raw scores for ROC Analysis
roc_data = {
    "y_true": [],
    "y_score": []
}

# Global variables 
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

# Results Saving Function 
def save_results(scores_data, topic_string):
    try:
        test_name = topic_string.split('/')[1] 
        
        # Save Summary Results
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
            
        print(f"\n\n*** Saved SUMMARY metrics to {filename} ***")
        print(json.dumps(results, indent=2))

        # Save ROC Raw Data
        roc_filename = f"roc_data_{test_name}.json"
        with open(roc_filename, "w") as f:
            json.dump(roc_data, f)
        print(f"Saved RAW SCORES for ROC curve to {roc_filename}")

    except Exception as e:
        print(f"Error saving results: {e}")


def on_connect(client, userdata, flags, rc, properties):
    global current_topic
    if rc == 0:
        print(f"Connected successfully to broker at {BROKER_ADDRESS}")
        client.subscribe(current_topic)
        print(f"Subscribed to topic: {current_topic}")
        
        if baseline_is_trained:
            print(f"Baseline is LOADED. Starting {current_mode} detection.")
        else:
            print(f"No baseline file found. Starting TRAINING (Residuals)...")
            print(f"Will collect {TRAINING_SAMPLES} samples...")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global baseline_is_trained, alarm_is_active, baseline
    global current_topic, current_mode, target_topic_after_training
    
    try:
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
        
        # Extract Values and Set points
        sensors = data.get("temperature_C", {})
        setpoints = data.get("setpoints", {})
        
        temp_val = data.get("temperature_C", {}).get("mean")
        ph_val = data.get("pH", {}).get("mean")
        rpm_val = data.get("rpm", {}).get("mean")
        
        temp_sp = setpoints.get("temperature_C")
        ph_sp = setpoints.get("pH")
        rpm_sp = setpoints.get("rpm")

        if None in [temp_val, ph_val, rpm_val, temp_sp, ph_sp, rpm_sp]:
            # Skip if any data is missing
            return

        # Calculate Errors
        temp_err = temp_val - temp_sp
        ph_err = ph_val - ph_sp
        rpm_err = rpm_val - rpm_sp
  
        # TRAINING LOGIC 
        if not baseline_is_trained:
            training_data["temperature"].append(temp_err)
            training_data["ph"].append(ph_err)
            training_data["rpm"].append(rpm_err)
            
            print(f"Collecting residuals... {len(training_data['temperature'])} / {TRAINING_SAMPLES}", end="\r")

            if len(training_data["temperature"]) >= TRAINING_SAMPLES:
                print("\nTRAINING COMPLETE")
                
                for sensor_name in training_data:
                    data_list = training_data[sensor_name]
                    baseline[sensor_name] = {
                        "mean": np.mean(data_list),
                        "std": np.std(data_list)
                    }
                    print(f"Baseline for '{sensor_name} Error':")
                    print(f"\nMean: {baseline[sensor_name]['mean']:.4f}")
                    print(f"\nStd Dev: {baseline[sensor_name]['std']:.4f}")
                
                save_baseline(baseline, BASELINE_FILE)
                
                baseline_is_trained = True
                old_topic = current_topic
                
                current_topic = target_topic_after_training
                current_mode = sys.argv[1].lower() 
                
                client.unsubscribe(old_topic)
                client.subscribe(current_topic)
                
                print(f"Baseline is LOADED. Starting {current_mode} detection.")
                
        # DETECTION LOGIC 
        else:
            # Calculate Z-Scores on the ERROR (not the raw value) Z = (CurrentError - MeanError) / StdDevError
            temp_z = (temp_err - baseline["temperature"]["mean"]) / baseline["temperature"]["std"]
            ph_z = (ph_err - baseline["ph"]["mean"]) / baseline["ph"]["std"]
            rpm_z = (rpm_err - baseline["rpm"]["mean"]) / baseline["rpm"]["std"]
            
            # Compute Pooled Score
            pooled_score = math.sqrt(temp_z**2 + ph_z**2 + rpm_z**2)
            
            # Save ROC Data
            faults_list = data.get("faults", {}).get("last_active", [])
            is_fault_present = len(faults_list) > 0
            roc_data["y_true"].append(1 if is_fault_present else 0)
            roc_data["y_score"].append(pooled_score)

            # Hysteresis Check
            if pooled_score > TAU_HIGH:
                alarm_is_active = True
            elif pooled_score < TAU_LOW:
                alarm_is_active = False
            
            # Keep Score
            if alarm_is_active and is_fault_present:
                scores["tp"] += 1
            elif alarm_is_active and not is_fault_present:
                scores["fp"] += 1
            elif not alarm_is_active and not is_fault_present:
                scores["tn"] += 1
            elif not alarm_is_active and is_fault_present:
                scores["fn"] += 1

            # Print Status
            alarm_status = "!! ALARM !!" if alarm_is_active else "Normal"
            truth_status = "FAULT" if is_fault_present else "Normal"
            
            print(f"Score: {pooled_score:.2f} | Err_T: {temp_err:.2f} Err_pH: {ph_err:.2f} | Status: {alarm_status} | Truth: {truth_status}", end="\n")

    except Exception as e:
        print(f"Error processing message: {e}")

# Program Logic 
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        sys.exit(1)
        
    current_mode = sys.argv[1].lower()
    
    if current_mode == "single":
        target_topic_after_training = TOPIC_TEST_SINGLE
    elif current_mode == "three":
        target_topic_after_training = TOPIC_TEST_THREE
    elif current_mode == "variable":
        target_topic_after_training = TOPIC_TEST_VARIABLE
    else:
        sys.exit(1)

    if not load_baseline(BASELINE_FILE):
        print(f"Warning: {BASELINE_FILE} not found. Running in TRAINING mode first.")
        print("Will automatically continue to detection after training.")
        current_topic = TOPIC_TRAINING
    else:
        current_topic = target_topic_after_training
    
    if not BROKER_ADDRESS:
        print("Error: BROKER_ADDRESS not found. Check your .env file.")
        sys.exit(1)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"group_5_{current_mode}")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
        sys.exit(1)

    print(f"Starting network loop. Mode: {current_mode}. Press CTRL+C to stop.")
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping detector...")
        if baseline_is_trained:
            save_results(scores, current_topic)
        
        client.disconnect()
        print("Disconnected from broker. Exiting.")
        sys.exit(0)