## PW1 - Lab A: Reproducible Foundations

**What I built:**
- Created a Conda environment with Python, NumPy and pytest.
- Implemented and tested radioactive decay simulations and compared the Python loop with NumPy.

**Speed comparison (loop vs NumPy):**
- loop: REPLACE_WITH_LOOP_TIME seconds
- numpy: REPLACE_WITH_NUMPY_TIME seconds
- speed-up: REPLACE_WITH_SPEEDUP x faster

**Tests:** all passing? REPLACE_WITH_YES_OR_NO

**Conclusion:**
- I learned how to manage a Python environment, test a simulation, and compare the performance of two implementations.
- The NumPy implementation was REPLACE_WITH_YOUR_OBSERVATION compared with the pure-Python loop.

## PW1 - Lab B: Data, Plotting, and Automation

**What I did:**
- Loaded the observed radioactive decay data from a CSV file.
- Compared the observed counts with the analytical decay law using two plots.
- Created a Snakemake pipeline to generate the figure automatically.

**Results:**
- The observed data and analytical law: REPLACE_WITH_YOUR_OBSERVATION.
- The generated figure is saved as `figure.png`.
- Snakemake successfully generated the figure and skipped unnecessary work on a subsequent run.

**Conclusion:**
- This practical helped me understand how to visualize experimental data and automate a data-processing workflow
<<<<<<< HEAD
=======
## PW2 Lab A: Motion from Tracking Data

### Results Summary
* **Mean Acceleration:** ~ -9.81 m/s² (confirms free-fall under gravity)[cite: 1, 2].
* **Acceleration Standard Deviation:** High (large fluctuations around the mean due to noise amplification)[cite: 2].
* **Maximum Position Difference:** < 1.0 m between original measured height and recovered position[cite: 2].

### Why is Acceleration So Noisy?
Differentiation acts as a high-pass filter that amplifies high-frequency measurement noise with each derivative step, making $a(t)$ significantly noisier than $y(t)$[cite: 2].

### Effect of Integration Back
Integrating acceleration back up to position suppresses and averages out the random measurement noise, successfully recovering the smooth trajectory close to the original coordinates[cite: 2].