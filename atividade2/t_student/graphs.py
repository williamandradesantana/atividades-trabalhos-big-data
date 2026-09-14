"""
Charts - Confidence Interval (95%) for Algorithm Execution Time
Student's t-distribution
"""

import numpy as np
from __init__ import data
import matplotlib.pyplot as plt
from scipy import stats
import os

n = len(data)
mean = np.mean(data)
std_dev = np.std(data, ddof=1)
std_error = std_dev / np.sqrt(n)
df = n - 1
confidence = 0.95
alpha = 1 - confidence
t_critical = stats.t.ppf(1 - alpha / 2, df)
margin_of_error = t_critical * std_error
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

# Figure with 2 side-by-side charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Chart 1: Data scatter + mean + CI
x_jitter = np.random.default_rng(42).normal(0, 0.04, size=n)
ax1.scatter(
    x_jitter, data, color="#4C72B0", alpha=0.7, s=50, zorder=3, label="Individual runs"
)

# Confidence interval band
ax1.axhspan(
    ci_lower,
    ci_upper,
    color="#DD8452",
    alpha=0.2,
    label=f"95% CI [{ci_lower:.2f}; {ci_upper:.2f}]",
)
ax1.axhline(
    mean, color="#DD8452", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f} ms"
)
ax1.axhline(ci_lower, color="#DD8452", linestyle=":", linewidth=1)
ax1.axhline(ci_upper, color="#DD8452", linestyle=":", linewidth=1)

ax1.set_xlim(-0.5, 0.5)
ax1.set_xticks([])
ax1.set_ylabel("Execution time (ms)")
ax1.set_title("Execution Times and 95% Confidence Interval")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Chart 2: Student's t-distribution with critical region
x = np.linspace(-4, 4, 500)
y = stats.t.pdf(x, df)

ax2.plot(x, y, color="#4C72B0", linewidth=2, label=f"Student's t (df = {df})")

# Central region (95% confidence)
x_central = np.linspace(-t_critical, t_critical, 300)
ax2.fill_between(
    x_central,
    stats.t.pdf(x_central, df),
    color="#55A868",
    alpha=0.3,
    label="95% confidence",
)

# Critical regions (tails)
x_left = np.linspace(-4, -t_critical, 100)
x_right = np.linspace(t_critical, 4, 100)
ax2.fill_between(x_left, stats.t.pdf(x_left, df), color="#C44E52", alpha=0.4)
ax2.fill_between(
    x_right,
    stats.t.pdf(x_right, df),
    color="#C44E52",
    alpha=0.4,
    label="Critical regions (2.5% each)",
)

ax2.axvline(-t_critical, color="#C44E52", linestyle="--", linewidth=1)
ax2.axvline(t_critical, color="#C44E52", linestyle="--", linewidth=1)
ax2.text(
    t_critical, max(y) * 0.55, f" t = {t_critical:.3f}", color="#C44E52", fontsize=9
)
ax2.text(
    -t_critical,
    max(y) * 0.55,
    f"t = {-t_critical:.3f} ",
    color="#C44E52",
    fontsize=9,
    ha="right",
)

ax2.set_xlabel("t value")
ax2.set_ylabel("Probability density")
ax2.set_title(f"Student's t-distribution (df = {df}) — 95% Critical Values")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "confidence_interval_charts.png")

plt.tight_layout()
plt.savefig(
   output_path, dpi=150, bbox_inches="tight"
)
print("Chart saved successfully!")
