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

    # -------------------- Step 1: Beacon stream --------------------
    beacon_file = config['log_file']
    if os.path.exists(beacon_file):
        print(f"Loading existing beacon stream from {beacon_file}")
        df = pd.read_csv(beacon_file)
    else:
        print("Generating beacon stream...")
        sim = BeaconSimulator(config)
        sim.initialize_devices()
        df = sim.run()
        df.to_csv(beacon_file, index=False)
        print(f"Saved {len(df)} advertisements to {beacon_file}")

    # -------------------- Step 2: Train anomaly detector --------------------
    normal_data = df[~df['is_rogue']].sample(frac=0.1)  # 10% of normal data
    X_train = normal_data[['rssi']].values
    edge = EdgeProcessor(config)
    edge.anomaly_detector.train(X_train)

    # -------------------- Step 3: Process advertisements with checkpointing --------------------
    print("Processing advertisements at edge...")
    df_sorted = df.sort_values('timestamp')

    # Checkpoint file
    checkpoint_interval = 10000  # save every 10,000 ads
    partial_file = os.path.join(config['results_dir'], 'partial_results.csv')

    results = []
    start_idx = 0

    # If a partial results file exists, load it and skip already processed rows
    if os.path.exists(partial_file):
        print(f"Found partial results file. Loading {partial_file} ...")
        existing = pd.read_csv(partial_file)
        results = existing.to_dict('records')
        start_idx = len(results)
        print(f"Resuming from advertisement index {start_idx} (out of {len(df_sorted)})")

    # Processing loop
    for idx, row in df_sorted.iloc[start_idx:].iterrows():
        adv = row.to_dict()
        out = edge.process_advertisement(adv)
        results.append(out)

        # Save checkpoint periodically
        if (len(results) % checkpoint_interval) == 0:
            pd.DataFrame(results).to_csv(partial_file, index=False)
            print(f"Checkpoint saved at {len(results)} advertisements")

    # -------------------- Step 4: Final evaluation and cleanup --------------------
    # Add predictions to the original dataframe
    df['pred_anomaly'] = [r['anomaly'] for r in results]
    df['true_anomaly'] = df['is_rogue']

    y_true = df['true_anomaly'].astype(int)
    y_pred = df['pred_anomaly'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Anomaly Detection: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Placeholder for session continuity evaluation
    print("Session continuity evaluation placeholder.")

    # Save full results
    final_file = os.path.join(config['results_dir'], 'processed_results.csv')
    df.to_csv(final_file, index=False)
    print(f"Full results saved to {final_file}")

    # Remove partial checkpoint file if it exists
    if os.path.exists(partial_file):
        os.remove(partial_file)
        print("Partial checkpoint file removed.")

if __name__ == "__main__":
    main()
