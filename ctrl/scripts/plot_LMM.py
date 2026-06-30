import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

def plot_synthetic_cases(df, title):
    """
    Plots the relationship between deltaZ and delta_precip for all cases,
    drawing a separate linear regression line for each case.
    """
    # lmplot automatically groups by 'case' and calculates regressions
    g = sns.lmplot(
        data=df,
        x='deltaZ_mean',
        y='delta_precip_along_beam',
        hue='case',
        ci=None, # Disable confidence bands to keep the plot readable
        scatter_kws={'alpha': 0.1, 's': 5}, # Make scatter points small and transparent
        line_kws={'linewidth': 2},
        height=6,
        aspect=1.5,
        palette='tab20'
    )

    plt.title(title)
    plt.subplots_adjust(top=0.9) # Prevent title clipping
    plt.show()

def generate_synthetic_radar_data(n_cases=17, samples_per_case=500, slope_variance=0.0):
    """
    Generates synthetic atmospheric data.
    If slope_variance > 0, the relationship between Z and Precip varies by case.
    If slope_variance == 0, the relationship is identical across all cases (only random noise varies).
    """
    # Seed for reproducibility so you get consistent output when testing
    np.random.seed(42)

    data = []
    base_slope = 0.5

    for case in range(n_cases):
        # Determine the physical relationship for this specific day
        # If variance is 0, every case gets exactly 0.5 as its true physical slope
        actual_case_slope = np.random.normal(base_slope, np.sqrt(slope_variance))
        actual_case_intercept = np.random.normal(0, 0.5)

        # Generate synthetic radar observations
        deltaZ_mean = np.random.normal(0, 1, samples_per_case)

        # Generate precipitation based on the day's specific slope + random measurement/sampling noise
        noise = np.random.normal(0, 1, samples_per_case)
        delta_precip = (actual_case_slope * deltaZ_mean) + actual_case_intercept + noise

        df = pd.DataFrame({
            'case': case,
            'deltaZ_mean': deltaZ_mean,
            'delta_precip_along_beam': delta_precip
        })
        data.append(df)

    return pd.concat(data, ignore_index=True)

def test_heterogeneity(df, title):
    """Runs the LMM Likelihood Ratio Test on the provided dataframe."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")

    # 1. Standardize variables to ensure readable variance components
    df = df.copy()
    df['precip_std'] = (df['delta_precip_along_beam'] - df['delta_precip_along_beam'].mean()) / df['delta_precip_along_beam'].std()
    df['deltaZ_std'] = (df['deltaZ_mean'] - df['deltaZ_mean'].mean()) / df['deltaZ_mean'].std()

    # 2. Fit Models
    m0 = smf.mixedlm("precip_std ~ deltaZ_std", df, groups=df["case"]).fit(reml=False)
    m1 = smf.mixedlm("precip_std ~ deltaZ_std", df, groups=df["case"], re_formula="~deltaZ_std").fit(reml=False)

    # 3. Calculate LRT
    lrt_stat = 2 * (m1.llf - m0.llf)
    df_diff = m1.df_modelwc - m0.df_modelwc
    p_val = chi2.sf(lrt_stat, df=df_diff)

    print(f"LRT Statistic: {lrt_stat:.2f}")
    print(f"p-value:       {p_val:.2e}")

    # Extract the estimated variance of the random slopes directly from the covariance matrix
    slope_var_estimate = m1.cov_re.iloc[1, 1]
    print(f"Estimated Slope Variance (deltaZ_std Var): {slope_var_estimate:.4f}")

    if p_val < 0.05:
        print("Verdict:       SIGNIFICANT. The day-to-day differences are real.")
    else:
        print("Verdict:       NOT SIGNIFICANT. Any differences are just sampling noise.")

# --- Run Scenario 1: True physical differences between days ---
# We inject a true variance of 0.2 into the slopes (e.g., convective vs. stratiform days)
df_heterogeneous = generate_synthetic_radar_data(slope_variance=0.2)
test_heterogeneity(df_heterogeneous, "SCENARIO 1: True Meteorological Differences")

# --- Run Scenario 2: No physical differences ---
# We force every single day to have the exact same underlying relationship.
# Individual cases will still have different r^2 values purely due to random sampling noise.
df_homogeneous = generate_synthetic_radar_data(slope_variance=0.0)
test_heterogeneity(df_homogeneous, "SCENARIO 2: Pure Sampling Noise")

# Add these calls to the bottom of the previous script:
plot_synthetic_cases(df_heterogeneous, "SCENARIO 1: True Differences (Heterogeneous)")
plot_synthetic_cases(df_homogeneous, "SCENARIO 2: Pure Sampling Noise (Homogeneous)")


