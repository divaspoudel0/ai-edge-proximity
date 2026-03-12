import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def generate_architecture_diagram():
    import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set high DPI and anti-aliasing
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors with gradients (light to slightly darker)
beacon_color = '#AED6F1'
beacon_edge = '#2E86C1'
edge_color = '#F9E79F'
edge_edge = '#B7950B'
cloud_color = '#F5B7B1'
cloud_edge = '#C0392B'

# Helper to create a rounded rectangle with gradient effect
def gradient_rectangle(ax, x, y, width, height, facecolor, edgecolor, linewidth=1.5, alpha=0.8, gradient_strength=0.2):
    # Base rectangle
    rect = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.2",
                           facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                           alpha=alpha)
    ax.add_patch(rect)
    # Add a subtle highlight at top for 3D effect
    highlight = Rectangle((x, y + height*0.9), width, height*0.1,
                          facecolor='white', edgecolor='none', alpha=0.2)
    ax.add_patch(highlight)
    # Add a subtle shadow at bottom
    shadow = Rectangle((x, y), width, height*0.1,
                       facecolor='black', edgecolor='none', alpha=0.05)
    ax.add_patch(shadow)

# ----- Layer 1: Beacon Layer (bottom) -----
beacon_y = 1.0
beacon_height = 1.8
beacon_width = 1.8
beacon_positions = [(2.0, beacon_y), (4.5, beacon_y), (7.0, beacon_y)]
beacon_labels = [
    'Beacon A\n(MAC, UID, SVC)',
    'Beacon B\n(MAC, UID, SVC)',
    'Beacon C\n(MAC, UID, SVC)'
]

for i, (x, y) in enumerate(beacon_positions):
    gradient_rectangle(ax, x, y, beacon_width, beacon_height,
                       facecolor=beacon_color, edgecolor=beacon_edge, alpha=0.9)
    ax.text(x + beacon_width/2, y + beacon_height/2, beacon_labels[i],
            ha='center', va='center', fontsize=9, fontweight='normal')

# Layer label
ax.text(5, beacon_y - 0.3, 'Beacon Layer', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#2C3E50')

# ----- Layer 2: AI Edge Layer (middle) -----
edge_y = 3.5
edge_height = 2.8
edge_width = 5.5
edge_x = 2.25

gradient_rectangle(ax, edge_x, edge_y, edge_width, edge_height,
                   facecolor=edge_color, edgecolor=edge_edge, alpha=0.95)

# Edge processor details (bullet points)
edge_text = (
    'AI Edge Processor\n'
    '• Contextual Fingerprinting\n'
    '• Anomaly Detection (Isolation Forest)\n'
    '• Intent Prediction (HMM)\n'
    '• Session Continuity Manager'
)
ax.text(edge_x + edge_width/2, edge_y + edge_height/2, edge_text,
        ha='center', va='center', fontsize=9, linespacing=1.5)

ax.text(5, edge_y - 0.3, 'AI Edge Layer', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#2C3E50')

# ----- Layer 3: Cloud Layer (top) -----
cloud_y = 7.0
cloud_height = 1.8
cloud_width = 4.5
cloud_x = 2.75

gradient_rectangle(ax, cloud_x, cloud_y, cloud_width, cloud_height,
                   facecolor=cloud_color, edgecolor=cloud_edge, alpha=0.95)

cloud_text = (
    'Cloud Coordination Server\n'
    '• Global identifier DB\n'
    '• Rotation schedules\n'
    '• Policy management'
)
ax.text(cloud_x + cloud_width/2, cloud_y + cloud_height/2, cloud_text,
        ha='center', va='center', fontsize=9)

ax.text(5, cloud_y - 0.3, 'Cloud Layer', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#2C3E50')

# ----- Arrows between layers -----
arrow_kwargs = dict(arrowstyle='->', mutation_scale=20, linewidth=1.5, color='#7F8C8D')

# From beacons to edge
for i, (x, y) in enumerate(beacon_positions):
    start = (x + beacon_width/2, y + beacon_height)
    end = (edge_x + edge_width/2, edge_y)
    arrow = FancyArrowPatch(start, end, **arrow_kwargs)
    ax.add_patch(arrow)

# From edge to cloud
start_edge = (edge_x + edge_width/2, edge_y + edge_height)
end_cloud = (cloud_x + cloud_width/2, cloud_y)
arrow_up = FancyArrowPatch(start_edge, end_cloud, **arrow_kwargs)
ax.add_patch(arrow_up)

# Dashed arrow from cloud back to edge (bidirectional communication)
arrow_down = FancyArrowPatch(end_cloud, start_edge,
                             arrowstyle='->', mutation_scale=20,
                             linewidth=1.5, color='#7F8C8D', linestyle='dashed')
ax.add_patch(arrow_down)

# Title
ax.set_title('AI-Augmented Edge Architecture for Proximity Services',
             fontsize=14, fontweight='bold', pad=20, color='#1A5276')

# Optional: very light grid lines for alignment (comment out if not wanted)
# for i in range(11):
#     ax.axhline(i, color='lightgray', linewidth=0.2, zorder=0)
#     ax.axvline(i, color='lightgray', linewidth=0.2, zorder=0)

plt.tight_layout()
plt.savefig('architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
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
