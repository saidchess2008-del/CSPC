"""
PW2 Lab B -- Part 4: Chemical Equilibrium H2 + I2 <-> 2HI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 16.0  # Equilibrium constant

def k_imbalance(x):
    return (2 * x)**2 / ((1 - x) * (1 - x)) - K

# 1. Newton's method root finding
x_newton = newton(k_imbalance, 0.5)

# 2. SLSQP minimization of squared imbalance
res = minimize(lambda x: k_imbalance(x[0])**2, x0=[0.5], method="SLSQP", bounds=[(0, 0.99)])
x_slsqp = res.x[0]

print(f"Equilibrium extent x (Newton): {x_newton:.4f}")
print(f"Equilibrium extent x (SLSQP):  {x_slsqp:.4f}")

h2_eq = 1 - x_newton
i2_eq = 1 - x_newton
hi_eq = 2 * x_newton

print(f"Equilibrium Amounts: [H2] = {h2_eq:.3f}, [I2] = {i2_eq:.3f}, [HI] = {hi_eq:.3f} mol")

# Plotting composition vs extent x
x_vals = np.linspace(0, 0.95, 200)
h2_vals = 1 - x_vals
i2_vals = 1 - x_vals
hi_vals = 2 * x_vals

plt.figure(figsize=(8, 6))
plt.plot(x_vals, h2_vals, label='$H_2$', color='blue')
plt.plot(x_vals, i2_vals, label='$I_2$', color='green')
plt.plot(x_vals, hi_vals, label='$HI$', color='red')
plt.axvline(x_newton, color='black', linestyle='--', label=f'Equilibrium $x\\approx{x_newton:.2f}$')
plt.xlabel('Extent of Reaction ($x$)')
plt.ylabel('Amount (mol)')
plt.title('Chemical Equilibrium Composition')
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png", dpi=300)
plt.show()