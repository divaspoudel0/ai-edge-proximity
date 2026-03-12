from sklearn.ensemble import IsolationForest
from hmmlearn import hmm
import numpy as np

class AnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state, warm_start=True)
        self.trained = False

    def train(self, X):
        """X: numpy array of shape (n_samples, n_features)"""
        self.model.fit(X)
        self.trained = True

    def predict(self, x):
        """x: feature vector (1D). Returns -1 for anomaly, 1 for normal."""
        if not self.trained:
            return 1
        return self.model.predict([x])[0]

    def score(self, x):
        """Anomaly score (lower = more anomalous)."""
        if not self.trained:
            return 0.5
        return self.model.decision_function([x])[0]

class IntentPredictor:
    def __init__(self, n_states=6, n_features=2):
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        self.n_features = n_features

    def train(self, observations, lengths=None):
        """observations: list of sequences or concatenated array with lengths."""
        self.model.fit(observations, lengths)

    def predict_next_state(self, recent_obs):
        """Given recent observation sequence, predict next state index."""
        # Use Viterbi to get most likely current state
        logprob, states = self.model.decode(recent_obs, algorithm="viterbi")
        current_state = states[-1]
        # Transition probabilities from current state
        next_state_probs = self.model.transmat_[current_state]
        next_state = np.argmax(next_state_probs)
        return next_state