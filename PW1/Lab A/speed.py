import time
from decay import simulate_loop, simulate

N0 = 200_000
lam = 0.4
dt = 0.05
steps = 200
seed = 0

# Pure Python version
start = time.perf_counter()
result_loop = simulate_loop(N0, lam, dt, steps, seed)
loop_time = time.perf_counter() - start

# NumPy version
start = time.perf_counter()
result_numpy = simulate(N0, lam, dt, steps, seed)
numpy_time = time.perf_counter() - start

print(f"Pure Python loop: {loop_time:.4f} seconds")
print(f"NumPy version:    {numpy_time:.4f} seconds")

if numpy_time > 0:
    print(f"NumPy is {loop_time / numpy_time:.2f}x faster")