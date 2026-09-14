"""
Confidence Interval (95%) for Algorithm Execution Time
Student's t-distribution - population standard deviation unknown

Requires: numpy, scipy (pip install scipy --break-system-packages, if needed)
"""

import numpy as np
from __init__ import data
from scipy import stats


n = len(data)
confidence = 0.95
alpha = 1 - confidence

# 2. Sample mean and sample standard deviation
mean = np.mean(data)
std_dev = np.std(data, ddof=1)
std_error = std_dev / np.sqrt(n)

# 3. Critical t value
df = n - 1
t_critical = stats.t.ppf(1 - alpha / 2, df)

# 4. Margin of error and Confidence Interval
margin_of_error = t_critical * std_error
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

# 5. Results
print(f"Sample size (n):                {n}")
print(f"Sample mean (x̄):                {mean:.4f} ms")
print(f"Sample standard deviation (s):  {std_dev:.4f} ms")
print(f"Standard error (s/√n):          {std_error:.4f} ms")
print(f"Degrees of freedom (n-1):       {df}")
print(f"Critical t value (95%, df={df}): {t_critical:.4f}")
print(f"Margin of error (E):            {margin_of_error:.4f} ms")
print(f"95% Confidence Interval:        ({ci_lower:.2f} ms ; {ci_upper:.2f} ms)")
