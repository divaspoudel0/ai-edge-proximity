import os
import pickle
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
    # We'll train only if we are not resuming, or if we need to train from scratch.
    # For simplicity, we always train; if resuming, the loaded processor will overwrite this.
    normal_data = df[~df['is_rogue']].sample(frac=0.1)  # 10% of normal data
    X_train = normal_data[['rssi']].values
    edge = EdgeProcessor(config)
    edge.anomaly_detector.train(X_train)

    # -------------------- Step 3: Check for existing checkpoints --------------------
    checkpoint_interval = 10000  # save every 10,000 ads
    partial_file = os.path.join(config['results_dir'], 'partial_results.csv')
    processor_checkpoint = os.path.join(config['results_dir'], 'edge_processor_checkpoint.pkl')

    results = []
    start_idx = 0

    # If both checkpoint files exist, load them and resume
    if os.path.exists(partial_file) and os.path.exists(processor_checkpoint):
        print("Found existing checkpoints. Resuming from last checkpoint...")
        # Load partial results
        existing_results = pd.read_csv(partial_file)
        results = existing_results.to_dict('records')
        start_idx = len(results)

        # Load the processor state
        with open(processor_checkpoint, 'rb') as f:
            edge = pickle.load(f)
        print(f"Loaded processor state. Resuming from advertisement index {start_idx} (out of {len(df)})")

    elif os.path.exists(partial_file) or os.path.exists(processor_checkpoint):
        # Inconsistent checkpoint files – warn and start fresh
        print("WARNING: Incomplete checkpoint files found. Starting fresh.")
        if os.path.exists(partial_file):
            os.remove(partial_file)
        if os.path.exists(processor_checkpoint):
            os.remove(processor_checkpoint)

    # -------------------- Step 4: Process advertisements with checkpointing --------------------
    print("Processing advertisements at edge...")
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    for idx, row in df_sorted.iloc[start_idx:].iterrows():
        adv = row.to_dict()
        out = edge.process_advertisement(adv)
        results.append(out)

        # Save checkpoint periodically
        if (len(results) % checkpoint_interval) == 0:
            # Save partial results
            pd.DataFrame(results).to_csv(partial_file, index=False)
            # Save processor state
            with open(processor_checkpoint, 'wb') as f:
                pickle.dump(edge, f)
            print(f"Checkpoint saved at {len(results)} advertisements")

    # -------------------- Step 5: Final evaluation and cleanup --------------------
    # Add predictions to the original dataframe
    # Note: results order matches df_sorted order.
    df_sorted['pred_anomaly'] = [r['anomaly'] for r in results]
    # Merge back to original df (which may have different order) – simpler: use df_sorted for evaluation
    # But original df has the ground truth. We'll create a new column aligned by index.
    # Since df_sorted is a sorted view, we can reindex to original order if needed.
    # For simplicity, we'll just evaluate on df_sorted.
    df_sorted['true_anomaly'] = df_sorted['is_rogue']

    y_true = df_sorted['true_anomaly'].astype(int)
    y_pred = df_sorted['pred_anomaly'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Anomaly Detection: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Placeholder for session continuity evaluation
    print("Session continuity evaluation placeholder.")

    # Save full results (using df_sorted to include predictions)
    final_file = os.path.join(config['results_dir'], 'processed_results.csv')
    df_sorted.to_csv(final_file, index=False)
    print(f"Full results saved to {final_file}")

    # Remove checkpoint files
    if os.path.exists(partial_file):
        os.remove(partial_file)
    if os.path.exists(processor_checkpoint):
        os.remove(processor_checkpoint)
    print("Checkpoint files removed.")

if __name__ == "__main__":
    main()
