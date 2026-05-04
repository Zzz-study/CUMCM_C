import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, probplot
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 1. Define GLMM Parameters (No Excel Dependency)
# --------------------------
params_data = {
    'param': ['beta0', 'beta_ga', 'beta_ga2', 'beta_bmi', 'sigma_u', 'phi'],
    'mean': [-2.519, 0.13, 0.03, -0.076, 0.349, 77.96],
    'sd': [0.03, 0.019, 0.014, 0.025, 0.022, 3.212],
    'hdi_3%': [-2.575, 0.097, 0.004, -0.122, 0.308, 72.198],
    'hdi_97%': [-2.464, 0.166, 0.056, -0.029, 0.392, 84.269]
}
df_params = pd.DataFrame(params_data)

# Extract parameters (compatible with NumPy 1.20+)
params = df_params['param'].values
means = df_params['mean'].values.astype(float)
sds = df_params['sd'].values.astype(float)
hdi_lower = df_params['hdi_3%'].values.astype(float)
hdi_upper = df_params['hdi_97%'].values.astype(float)

# Generate input data for prediction
np.random.seed(42)
ga_range = np.linspace(10, 40, 100)  # Gestational Age (weeks: 10-40)
bmi_range = np.linspace(18, 35, 5)   # BMI (18-35: normal to obese)
cov_matrix = np.diag(sds ** 2)       # Parameter covariance matrix


# --------------------------
# 2. Plot Style Configuration (High-Resolution + English Font)
# --------------------------
def set_plot_style():
    plt.rcParams['font.sans-serif'] = ['Arial']  # English font (avoid Chinese issues)
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    plt.rcParams['savefig.dpi'] = 300  # High resolution for saving
    plt.rcParams['figure.facecolor'] = 'white'  # White background (no transparent blank)

set_plot_style()


# --------------------------
# Figure 1: Parameter Estimates with 97% HDI
# --------------------------
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(params))

# Plot mean and HDI intervals
ax.scatter(means, y_pos, color='#2E86AB', s=80, zorder=3, label='Parameter Mean')
ax.hlines(y_pos, hdi_lower, hdi_upper, color='#A23B72', linewidth=3, zorder=2, label='97% HDI')
ax.vlines(0, -1, len(params), color='#F18F01', linestyle='--', alpha=0.7, zorder=1, label='Reference Line (Value=0)')

# Annotations
ax.set_yticks(y_pos)
ax.set_yticklabels(params, fontsize=11)
ax.set_xlabel('Parameter Estimate', fontsize=12, labelpad=10)
ax.set_title('GLMM Parameter Estimates and 97% Highest Density Intervals (HDI)', fontsize=14, pad=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)

# Save
plt.tight_layout()
plt.savefig('glmm_param_hdi_en.png', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("✅ Figure 1 saved: glmm_param_hdi_en.png")


# --------------------------
# Figure 2: Model Prediction Curves (GA × BMI)
# --------------------------
# GLMM prediction function (logistic regression inverse transform)
def glmm_predict(ga, bmi, beta0, beta_ga, beta_ga2, beta_bmi):
    linear_pred = beta0 + beta_ga * ga + beta_ga2 * (ga ** 2) + beta_bmi * bmi
    return 1 / (1 + np.exp(-linear_pred))  # Probability output (0-1)

# Parameter sampling for 95% confidence interval
n_samples = 1000
param_samples = multivariate_normal.rvs(mean=means, cov=cov_matrix, size=n_samples)
pred_probs = np.zeros((len(ga_range), len(bmi_range), n_samples))

# Calculate predicted probabilities for each BMI
for i, bmi in enumerate(bmi_range):
    for j, sample in enumerate(param_samples):
        beta0, beta_ga, beta_ga2, beta_bmi, _, _ = sample
        pred_probs[:, i, j] = glmm_predict(ga_range, bmi, beta0, beta_ga, beta_ga2, beta_bmi)

# Compute mean and 95% CI
pred_mean = np.mean(pred_probs, axis=2)
pred_lower = np.percentile(pred_probs, 2.5, axis=2)
pred_upper = np.percentile(pred_probs, 97.5, axis=2)

# Plot curves
fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']  # BMI group colors

for i, (bmi, color) in enumerate(zip(bmi_range, colors)):
    ax.plot(ga_range, pred_mean[:, i], color=color, linewidth=2.5, label=f'BMI={bmi:.1f}')
    ax.fill_between(ga_range, pred_lower[:, i], pred_upper[:, i], color=color, alpha=0.15)

# Annotations
ax.set_xlabel('Gestational Age (weeks)', fontsize=12)
ax.set_ylabel('Predicted Probability (e.g., Event Probability)', fontsize=12)
ax.set_title('Predicted Probability by Gestational Age (Different BMI Groups)', fontsize=14, pad=15, fontweight='bold')
ax.legend(title='BMI Groups', loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(10, 40)
ax.set_ylim(0, 1)  # Probability range: 0-1

# Save
plt.tight_layout()
plt.savefig('glmm_prediction_curves_en.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Figure 2 saved: glmm_prediction_curves_en.png")


# --------------------------
# Figure 3: Parameter Correlation Heatmap
# --------------------------
param_sample_df = pd.DataFrame(param_samples, columns=params)
corr_matrix = param_sample_df.corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Red-blue colormap

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    center=0,
    square=True,
    linewidths=0.5,
    annot=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.8},
    ax=ax
)

# Annotations
ax.set_title('GLMM Parameter Correlation Heatmap (1000 Samples)', fontsize=14, pad=15, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Save
plt.tight_layout()
plt.savefig('glmm_param_correlation_en.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Figure 3 saved: glmm_param_correlation_en.png")


# --------------------------
# Figure 4: Model Diagnostic Plots
# --------------------------
n_obs = 500
# Simulate random effects (u_i ~ N(0, sigma_u²))
sigma_u = means[params == 'sigma_u'][0]
random_effects = np.random.normal(0, sigma_u, n_obs)

# Simulate fitted values and residuals
sim_ga = np.random.uniform(10, 40, n_obs)
sim_bmi = np.random.uniform(18, 35, n_obs)
sim_linear_pred = means[0] + means[1]*sim_ga + means[2]*(sim_ga**2) + means[3]*sim_bmi + random_effects
sim_fitted = 1 / (1 + np.exp(-sim_linear_pred))  # Fitted probabilities
sim_residuals = np.random.normal(0, 0.15, n_obs)  # Standardized residuals

# 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Subplot 1: Residuals vs Fitted Values
axes[0].scatter(sim_fitted, sim_residuals, alpha=0.6, color='#3498DB', s=30)
axes[0].axhline(y=0, color='#E74C3C', linestyle='--')
axes[0].set_xlabel('Fitted Values', fontsize=10)
axes[0].set_ylabel('Standardized Residuals', fontsize=10)
axes[0].set_title('Residuals vs Fitted Values', fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.3)

# Subplot 2: Residual Q-Q Plot
probplot(sim_residuals, dist='norm', plot=axes[1])
axes[1].set_title('Residual Normal Q-Q Plot', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.3)

# Subplot 3: Random Effects Distribution
axes[2].hist(random_effects, bins=20, density=True, alpha=0.6, color='#2ECC71', edgecolor='black')
x_norm = np.linspace(random_effects.min(), random_effects.max(), 100)
y_norm = (1 / (sigma_u * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_norm / sigma_u) ** 2)
axes[2].plot(x_norm, y_norm, color='#F39C12', linewidth=2)
axes[2].set_xlabel('Random Intercepts', fontsize=10)
axes[2].set_ylabel('Density', fontsize=10)
axes[2].set_title('Random Effects Distribution (Theoretical Normal)', fontsize=11, fontweight='bold')
axes[2].grid(alpha=0.3)

# Subplot 4: Random Effects Q-Q Plot
probplot(random_effects, dist='norm', plot=axes[3])
axes[3].set_title('Random Effects Normal Q-Q Plot', fontsize=11, fontweight='bold')
axes[3].grid(alpha=0.3)

# Main title
fig.suptitle('GLMM Model Diagnostic Plots (Simulated Data)', fontsize=14, y=0.98, fontweight='bold')

# Save
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('glmm_diagnostics_en.png', bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Figure 4 saved: glmm_diagnostics_en.png")
print("\n🎉 All English-labeled figures generated successfully!")