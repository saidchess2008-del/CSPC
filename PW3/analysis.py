import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# DATASET 1: HEART DISEASE (CLEVELAND)
# ==========================================
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print(df.head(), "\n")

cont_vars = ["age", "chol", "trestbps", "thalach"]
print("--- Summary Statistics ---")
print(df[cont_vars].describe(), "\n")

# --- Task 2: Visualise Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(cont_vars):
    axes[i].hist(df[col], bins=15, edgecolor="black", alpha=0.75, color="skyblue")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

# --- Task 3: Normality Checks ---
print("--- Task 3: Normality Tests (Shapiro-Wilk) ---")
for col in cont_vars:
    stat, p = stats.shapiro(df[col].dropna())
    skew = stats.skew(df[col].dropna())
    verdict = "Approximately Normal" if p > 0.05 else "Non-Normal"
    print(f"{col:10s} | W = {stat:.4f}, p = {p:.4e}, skew = {skew:+.4f} -> {verdict}")

# --- Task 4: Group Comparison (thalach by target) ---
healthy = df[df["target"] == 0]["thalach"]
disease = df[df["target"] == 1]["thalach"]

mw_stat, mw_p = stats.mannwhitneyu(healthy, disease)
t_stat, t_p = stats.ttest_ind(healthy, disease, equal_var=False)

print("\n--- Task 4: Max Heart Rate (thalach) by Target ---")
print(f"Healthy (0): n={len(healthy)}, Mean = {healthy.mean():.2f} ± {healthy.sem():.2f} (SEM)")
print(f"Disease (1): n={len(disease)}, Mean = {disease.mean():.2f} ± {disease.sem():.2f} (SEM)")
print(f"Mann-Whitney U = {mw_stat:.1f}, p = {mw_p:.4e}")
print(f"Welch's t-test = {t_stat:.4f}, p = {t_p:.4e}")

# Plot Task 4
plt.figure(figsize=(7, 5))
groups = ["Healthy (target=0)", "Heart Disease (target=1)"]
means = [healthy.mean(), disease.mean()]
sems = [healthy.sem(), disease.sem()]
bars = plt.bar(groups, means, yerr=sems, capsize=8, color=["steelblue", "salmon"], edgecolor="black", width=0.5)
plt.ylabel("Maximum Heart Rate (thalach, bpm)")
plt.title("Mean Maximum Heart Rate ± 1 SEM by Heart Disease Status")
plt.ylim(0, 185)
for bar, mean, sem in zip(bars, means, sems):
    plt.text(bar.get_x() + bar.get_width()/2, mean + sem + 4, f"{mean:.1f} ± {sem:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("thalach_comparison.png")
plt.close()

# --- Task 5: Age vs. Max Heart Rate Correlation ---
r_p, p_p = stats.pearsonr(df["age"], df["thalach"])
r_s, p_s = stats.spearmanr(df["age"], df["thalach"])
print("\n--- Task 5: Age vs. Thalach Correlation ---")
print(f"Pearson  r = {r_p:.4f} (p = {p_p:.4e})")
print(f"Spearman ρ = {r_s:.4f} (p = {p_s:.4e})")

# Plot Task 5
plt.figure(figsize=(8, 6))
plt.scatter(df["age"], df["thalach"], alpha=0.6, color="teal", edgecolors="k", label="Patients")
m, b = np.polyfit(df["age"], df["thalach"], 1)
plt.plot(df["age"], m * df["age"] + b, color="crimson", linewidth=2, label=f"Linear Fit (r = {r_p:.2f})")
plt.title("Age vs. Maximum Heart Rate (thalach)")
plt.xlabel("Age (years)")
plt.ylabel("Maximum Heart Rate (thalach, bpm)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("age_vs_thalach.png")
plt.close()

# --- Bonus: Categorical Unpredictability (Shannon Entropy) ---
props = df["target"].value_counts(normalize=True)
entropy = stats.entropy(props, base=2)
print("\n--- Bonus: Target Balance & Entropy ---")
print(props)
print(f"Shannon Entropy: {entropy:.4f} bits (Max possible for binary = 1.0000 bit)")


# ==========================================
# DATASET 2: CHEMICAL-EXPOSURE MYSTERY
# ==========================================
df_chem = pd.read_csv("chemicals_cancer.csv")

# --- Task 6: Naive Analysis ---
print("\n--- Task 6: Naive Correlations with Malignancy ---")
for col in ["benzene", "cadmium", "pollution_index", "age"]:
    r, p = stats.pearsonr(df_chem[col], df_chem["malignancy"])
    print(f"{col:16s} | r = {r:+.4f}, p = {p:.4e}")

# --- Task 7: Controlling for Confounder (pollution_index) ---
# Restrict to patients with similar pollution_index (e.g., 45 < pollution_index < 55)
subset = df_chem[(df_chem["pollution_index"] > 45) & (df_chem["pollution_index"] < 55)]
r_benz_sub, p_benz_sub = stats.pearsonr(subset["benzene"], subset["malignancy"])
r_cad_sub, p_cad_sub = stats.pearsonr(subset["cadmium"], subset["malignancy"])

print(f"\n--- Task 7: Stratified Analysis (45 < pollution_index < 55, n={len(subset)}) ---")
print(f"Benzene vs. Malignancy | r = {r_benz_sub:+.4f}, p = {p_benz_sub:.4e}")
print(f"Cadmium vs. Malignancy | r = {r_cad_sub:+.4f}, p = {p_cad_sub:.4e}")