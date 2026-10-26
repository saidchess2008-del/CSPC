"""
PW2 Lab B -- Part 2: Three routes to a minimum
- Gradient descent by hand
- scipy.optimize.newton on f'(x)
- scipy.optimize.minimize with SLSQP
"""

import numpy as np
from scipy.optimize import minimize, newton

# --- 2A. Easy convex function f(x) = (x - 3)**2 + 1 ---
print("=== 2A. Easy Convex Function ===")

def f(x):
    return (x - 3)**2 + 1

def f_prime(x):
    return 2 * (x - 3)

# 1. Gradient descent by hand
x = 0.0
lr = 0.1
for _ in range(100):
    x -= lr * f_prime(x)
print(f"Gradient Descent result: x = {x:.4f}")

# 2. Newton's method on f'(x) = 0 (f''(x) = 2)
x_newton = newton(f_prime, x0=0.0)
print(f"Newton's method result:  x = {x_newton:.4f}")

# 3. SLSQP minimize
res_slsqp = minimize(f, x0=[0.0], method="SLSQP")
print(f"SLSQP minimize result:   x = {res_slsqp.x[0]:.4f}")


# --- 2B. Harder landscape g(x) = x**4 - 3*x**2 + x + 5 ---
print("\n=== 2B. Harder Landscape ===")

def g(x):
    return x**4 - 3*x**2 + x + 5

def g_prime(x):
    return 4*x**3 - 6*x + 1

def g_prime2(x):
    return 12*x**2 - 6

for x0 in [0.0, 2.0]:
    print(f"\n--- Starting from x0 = {x0} ---")
    
    # Gradient descent by hand for g(x)
    x = x0
    for _ in range(1000):
        x -= 0.01 * g_prime(x)
    print(f"Gradient Descent: x = {x:.4f}, g(x) = {g(x):.4f}")
    
    # Newton's method on g'(x) = 0
    try:
        x_n = newton(g_prime, x0=x0, fprime=g_prime2)
        curvature = g_prime2(x_n)
        nature = "minimum" if curvature > 0 else "maximum"
        print(f"Newton's method:  x = {x_n:.4f} (g''(x) = {curvature:.2f} -> {nature})")
    except Exception as e:
        print(f"Newton's method failed: {e}")
        
    # SLSQP minimize
    res_g = minimize(g, x0=[x0], method="SLSQP")
    print(f"SLSQP minimize:   x = {res_g.x[0]:.4f}, g(x) = {res_g.fun:.4f}")