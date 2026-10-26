"""
PW2 Lab B -- Part 5: Titration Equivalence Point
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titration.csv")
V = df["volume_base"].values
pH = df["pH"].values

# Compute slope (derivative of pH w.r.t V)
slope = np.gradient(pH, V)
max_idx = np.argmax(slope)
eq_volume = V[max_idx]

print(f"Equivalence point volume: {eq_volume:.2f} mL")

# Plotting side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(V, pH, color='purple')
ax1.axvline(eq_volume, color='red', linestyle='--', label=f'Equivalence Point ({eq_volume:.1f} mL)')
ax1.set_xlabel('Base Volume (mL)')
ax1.set_ylabel('pH')
ax1.set_title('Titration Curve')
ax1.legend()
ax1.grid(True)

ax2.plot(V, slope, color='orange')
ax2.axvline(eq_volume, color='red', linestyle='--', label='Peak Slope')
ax2.set_xlabel('Base Volume (mL)')
ax2.set_ylabel('Slope ($dpH / dV$)')
ax2.set_title('Derivative (Slope) Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("titration.png", dpi=300)
plt.show()