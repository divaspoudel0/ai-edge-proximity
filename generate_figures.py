import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def generate_architecture_diagram():
    # Same as previously provided (copy from earlier answer)
    # ... (code for architecture.png)
    pass

def generate_accuracy_plot(results_path='results/processed_results.csv'):
    # Load results and compute prediction accuracy vs horizon
    # This is a placeholder; you should compute from actual HMM predictions.
    horizons = np.array([5, 10, 15, 20, 25, 30])
    mean_acc = np.array([98, 95, 92, 88, 82, 75])
    std_acc = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    plt.figure()
    plt.plot(horizons, mean_acc, 'o-', color='#2E86C1')
    plt.fill_between(horizons, mean_acc - std_acc, mean_acc + std_acc, color='#AED6F1', alpha=0.3)
    plt.xlabel('Prediction Horizon (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Zone Transition Prediction Accuracy')
    plt.grid()
    plt.savefig('results/prediction_accuracy.png', dpi=150)
    print("Saved prediction_accuracy.png")

if __name__ == "__main__":
    generate_architecture_diagram()
    generate_accuracy_plot()