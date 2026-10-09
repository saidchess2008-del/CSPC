import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

# Read observed data from CSV, skipping the header
data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)

t = data[:, 0]
observed = data[:, 1]

# Analytical decay law
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# Create two plots with shared axes
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                         figsize=(12, 5))

# Left: observed data
axes[0].scatter(t, observed)
axes[0].set_title("Observed radioactive decay")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Number of atoms")

# Right: analytical prediction
axes[1].plot(t, analytical)
axes[1].set_title("Analytical decay law")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Number of atoms")

fig.tight_layout()
fig.savefig("figure.png", dpi=150)
plt.close(fig)

print("Saved figure.png")