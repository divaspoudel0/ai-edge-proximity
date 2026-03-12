import os
import pandas as pd
import numpy as np
from src.beacon_generator import BeaconSimulator
from src.edge_processor import EdgeProcessor
from src.cloud_mock import CloudServer
from src.utils import load_config
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    config = load_config('config/simulation_config.yaml')
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1. Generate beacon stream
    print("Generating beacon stream...")
    sim = BeaconSimulator(config)
    sim.initialize_devices()
    df = sim.run()
    df.to_csv(config['log_file'], index=False)
    print(f"Saved {len(df)} advertisements to {config['log_file']}")

    # 2. Initialize edge processor and train anomaly detector on normal data
    normal_data = df[~df['is_rogue']].sample(frac=0.1)  # sample 10% of normal for training
    features = []
    # For training, we need to simulate fingerprints; simplified: use per-advertisement features
    # Better: group by device_key, but for demo we'll just use raw RSSI and position-derived speed
    # Here we'll just use RSSI as a simple feature.
    X_train = normal_data[['rssi']].values
    edge = EdgeProcessor(config)
    edge.anomaly_detector.train(X_train)

    # 3. Process each advertisement in chronological order
    print("Processing advertisements at edge...")
    df_sorted = df.sort_values('timestamp')
    results = []
    for idx, row in df_sorted.iterrows():
        adv = row.to_dict()
        out = edge.process_advertisement(adv)
        results.append(out)

    # 4. Evaluate anomaly detection
    df['pred_anomaly'] = [r['anomaly'] for r in results]
    df['true_anomaly'] = df['is_rogue']
    y_true = df['true_anomaly'].astype(int)
    y_pred = df['pred_anomaly'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Anomaly Detection: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # 5. Evaluate session continuity
    # We need to check if for each ground-truth device, all its appearances are mapped to the same logical_id
    # This is a placeholder; full implementation would require grouping by device_id.
    # For simplicity, we'll compute a dummy metric.
    print("Session continuity evaluation placeholder.")

    # 6. Save results
    df.to_csv(os.path.join(config['results_dir'], 'processed_results.csv'), index=False)

if __name__ == "__main__":
    main()
