import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_static_positions(num_beacons, width, height):
    return np.random.rand(num_beacons, 2) * [width, height]

def rssi_from_distance(d, P0=-59, n=2.0):
    """Simple path-loss model: RSSI = P0 - 10*n*log10(d)"""
    if d < 0.01:
        d = 0.01
    return P0 - 10 * n * np.log10(d)

def compute_similarity(fp1, fp2):
    """Cosine similarity between feature vectors (example)."""
    # Features: [mean_rssi, rssi_std, movement_speed, ...]
    # Normalize first.
    v1 = np.array(fp1.get_features())
    v2 = np.array(fp2.get_features())
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)