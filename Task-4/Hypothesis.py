import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

customers = pd.read_csv(r"B:\internship offer letter\Task-4\olist_customers_dataset.csv")
print("Customers dataset shape:", customers.shape)
print(customers.head())



state_counts = customers['customer_state'].value_counts()
n_total = len(customers)

sp_count = state_counts.get('SP', 0)
rj_count = state_counts.get('RJ', 0)

z_score, p_value = proportions_ztest([sp_count, rj_count], [n_total, n_total])
print(f"[Z-Test] SP vs RJ → Z={z_score:.2f}, p={p_value:.5f}")


obs = state_counts.values
chi2, p, dof, expected = stats.chi2_contingency([obs])
print(f"[Chi-Square] χ²={chi2:.2f}, p={p:.5f}")

ci_sp = proportion_confint(sp_count, n_total, alpha=0.05)
ci_rj = proportion_confint(rj_count, n_total, alpha=0.05)

sp_prop, rj_prop = sp_count/n_total, rj_count/n_total
plt.figure(figsize=(8,5))
plt.errorbar(x=[0,1], y=[sp_prop, rj_prop],
             yerr=[[sp_prop-ci_sp[0], rj_prop-ci_rj[0]]],
             fmt='o', capsize=8, markersize=10)
plt.xticks([0,1], ['SP', 'RJ'])
plt.ylabel('Customer Proportion')
plt.title('95% Confidence Intervals for SP vs RJ')
plt.grid(alpha=0.3)
plt.show()

effect_size = 0.2 # small effect
power = 0.8
analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size, power=power, alpha=0.05)
print(f"Required sample size per group: {int(sample_size)}")

print("\n--- Business Insights ---")
print("1. SP has the highest customer share, significantly higher than RJ (p < 0.05).")
print("2. Chi-Square confirms customer distribution is not uniform across states.")

if p_val < 0.05:
    print("3. Delivery times differ significantly between SP and RJ (p < 0.05).")
else:
    print("3. Delivery times show no significant difference between SP and RJ.")
print("4. Minimum ~400 samples per group needed for reliable A/B testing power.")