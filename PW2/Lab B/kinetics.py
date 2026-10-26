"""
PW2 Lab B -- Part 3: Fit a reaction rate constant k
C(t) = C0 * exp(-k * t)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Read kinetics data
df = pd.read_csv("kinetics.csv")
t = df["time"].values
C_meas = df["concentration"].values
C0 = C_meas[0]

def total_error(k):
    k_val = k[0] if hasattr(k, "__len__") else k
    pred = C0 * np.exp(-k_val * t)
    return np.sum((C_meas - pred)**2)

# Minimize error with SLSQP
res = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]
print(f"Fitted rate constant k = {fitted_k:.4f}")

# Plotting
t_smooth = np.linspace(t.min(), t.max(), 200)
C_fit = C0 * np.exp(-fitted_k * t_smooth)

plt.figure(figsize=(8, 6))
plt.scatter(t, C_meas, color='red', label='Measured Data')
plt.plot(t_smooth, C_fit, color='blue', label=f'Fitted Curve ($k={fitted_k:.3f}$)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration ($C$)')
plt.title('First-Order Reaction Kinetics Fit')
plt.legend()
plt.grid(True)
plt.savefig("kinetics.png", dpi=300)
plt.show()