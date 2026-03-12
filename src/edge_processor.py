import numpy as np
from collections import defaultdict
from .models import AnomalyDetector, IntentPredictor
from .utils import compute_similarity

class Fingerprint:
    """Tracks a device's recent history."""
    def __init__(self, device_key, init_adv):
        self.device_key = device_key  # e.g., (mac, uid, service_id)
        self.first_seen = init_adv['timestamp']
        self.last_seen = init_adv['timestamp']
        self.rssi_history = [init_adv['rssi']]
        self.positions = [(init_adv['x'], init_adv['y'])]  # for evaluation only
        self.service_id = init_adv['service_id']
        self.mac = init_adv['mac']
        self.uid = init_adv['uid']
        self.state_history = []  # for HMM observations

    def update(self, adv):
        self.last_seen = adv['timestamp']
        self.rssi_history.append(adv['rssi'])
        if len(self.rssi_history) > 50:
            self.rssi_history.pop(0)
        self.positions.append((adv['x'], adv['y']))

    def get_features(self):
        """Extract feature vector for anomaly detection."""
        if len(self.rssi_history) < 2:
            mean_rssi = self.rssi_history[-1] if self.rssi_history else -100
            std_rssi = 0
        else:
            mean_rssi = np.mean(self.rssi_history)
            std_rssi = np.std(self.rssi_history)
        # Additional features: time since last seen, movement speed (approx), etc.
        time_since_first = self.last_seen - self.first_seen
        # Approx speed from positions (if available)
        speed = 0
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[0][0]
            dy = self.positions[-1][1] - self.positions[0][1]
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / time_since_first if time_since_first > 0 else 0
        return np.array([mean_rssi, std_rssi, speed])

class EdgeProcessor:
    def __init__(self, config):
        self.config = config
        self.fingerprints = {}  # device_key -> Fingerprint
        self.anomaly_detector = AnomalyDetector(contamination=config['anomaly_contamination'])
        self.intent_predictor = IntentPredictor(n_states=config['hmm_n_states'])
        self.session_manager = SessionManager(config['similarity_threshold'])
        self.trained_anomaly = False

    def train_anomaly_detector(self, normal_data):
        """normal_data: list of feature vectors from benign devices."""
        X = np.array(normal_data)
        self.anomaly_detector.train(X)
        self.trained_anomaly = True

    def process_advertisement(self, adv):
        # 1. Update fingerprint
        key = (adv['mac'], adv['uid'], adv['service_id'])
        if key not in self.fingerprints:
            self.fingerprints[key] = Fingerprint(key, adv)
        else:
            self.fingerprints[key].update(adv)

        fp = self.fingerprints[key]
        features = fp.get_features()

        # 2. Anomaly detection
        is_anomaly = False
        if self.trained_anomaly:
            pred = self.anomaly_detector.predict(features)
            if pred == -1:
                is_anomaly = True
                # Could trigger alert, but for now just flag

        # 3. Intent prediction (if we have enough history)
        predicted_next = None
        if len(fp.state_history) >= 5:
            # Prepare observation sequence (e.g., RSSI and maybe zone indices)
            obs = np.array(fp.state_history[-10:]).reshape(-1, 1)  # simple: just RSSI
            predicted_next = self.intent_predictor.predict_next_state(obs)

        # 4. Session continuity
        logical_device = self.session_manager.link(fp, predicted_next, adv)

        return {
            'key': key,
            'anomaly': is_anomaly,
            'predicted_next': predicted_next,
            'logical_device': logical_device
        }

class SessionManager:
    def __init__(self, similarity_thresh=0.8):
        self.similarity_thresh = similarity_thresh
        self.logical_devices = {}  # logical_id -> set of keys
        self.key_to_logical = {}   # key -> logical_id

    def link(self, fingerprint, prediction, adv):
        # Try to find existing logical device that this fingerprint might belong to
        best_match = None
        best_sim = 0
        for lid, keys in self.logical_devices.items():
            # For each key in that logical device, get its fingerprint
            for k in keys:
                if k in edge_processor.fingerprints:  # need access to edge_processor? This is messy.
                    # Simplified: compare feature similarity
                    other_fp = edge_processor.fingerprints[k]
                    sim = compute_similarity(fingerprint, other_fp)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = lid
        if best_sim > self.similarity_thresh:
            # Assign to existing logical device
            lid = best_match
            self.logical_devices[lid].add(fingerprint.device_key)
            self.key_to_logical[fingerprint.device_key] = lid
        else:
            # Create new logical device
            lid = len(self.logical_devices)
            self.logical_devices[lid] = {fingerprint.device_key}
            self.key_to_logical[fingerprint.device_key] = lid
        return lid