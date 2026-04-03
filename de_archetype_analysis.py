import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity

# Load the dataset
data = pd.read_csv('/Users/bendoyle/Library/CloudStorage/OneDrive-TexasA&MUniversity/BD-JDT-TAMU-Sport-Data-Challenge-2026/Post-Query Datasets/PRE_NFL_DATA.csv')

#---------------------------------------
# Feature selection
#---------------------------------------

college_variables = ['COLLEGE_SACKS','COLLEGE_FORCED_FUMBLES',
                             'COLLEGE_TACKLES_SOLO','COLLEGE_TACKLES_ASSISTED',
                             'COLLEGE_PASSES_DEFENDED',
                             'COLLEGE_INTERCEPTIONS']
anthropometric_variables = ['HEIGHT_IN','WEIGHT_LB']

#---------------------------------------
# Data preprocessing
#---------------------------------------

# Step 1: Filter players with more than 3 missing combine tests using COMBINE_TESTS_MISSING from the SQL data
print(f"Total players before filtering: {data.shape[0]}")
data = data[data['COMBINE_TESTS_MISSING'] <= 3].copy()
print(f"Players after filtering (≤3 missing combine tests): {data.shape[0]}")

# Step 2: Compute velocity variables from time-based combine variables (after filtering)
data['X40Y_VEL'] = 36.57 / data['X40Y_SEC']
data['X3CD_VEL'] = 27.43 / data['X3CD_SEC']
data['PROA_VEL'] = 18.28 / data['PROA_SEC']
data['mCODD'] = data['PROA_SEC']/data['X40Y_SEC']  

# Step 3: Define combine variables for analysis — velocity replaces time; no time-based variables used in factor analysis
combine_variables = ['VJ_IN','SLJ_IN','BENCH_REPS','P40', 'MANN_SLJP','X40Y_VEL','X3CD_VEL','PROA_VEL']

# Step 4: Impute remaining missing values using Random Forest (combine + anthropometric only — no college variables)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def random_forest_imputation(df, target_col, feature_cols):
    known = df[df[target_col].notnull()]
    unknown = df[df[target_col].isnull()]
    
    if unknown.empty:
        return df
    
    X_train = known[feature_cols]
    y_train = known[target_col]
    X_unknown = unknown[feature_cols]
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_unknown_imputed = imputer.transform(X_unknown)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)
    
    predicted_values = model.predict(X_unknown_imputed)
    df.loc[df[target_col].isnull(), target_col] = predicted_values
    
    return df

for var in combine_variables:
    data = random_forest_imputation(data, var, [col for col in combine_variables if col != var] + anthropometric_variables)

# Standardize and center the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[combine_variables + college_variables + anthropometric_variables])

#---------------------------------------
# Exploratory Factor Analysis w/ oblique rotation using FactorAnalyzer package, optimizing number of factors based on eigenvalues > 1 and scree plot
#---------------------------------------

# Determine the number of factors using eigenvalues
fa = FactorAnalyzer(rotation=None)
fa.fit(data_scaled)
eigenvalues, _ = fa.get_eigenvalues()
# Plot the scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')
plt.grid()
plt.show()

# Select # of factors based on eigenvalues > 1 and scree plot
n_factors = 4
fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
fa.fit(data_scaled)

# Get factor loadings
loadings = fa.loadings_
# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, index=combine_variables + college_variables + anthropometric_variables, columns=[f'Factor_{i+1}' for i in range(n_factors)])
print(loadings_df)  

# Visualize factor loadings with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
plt.title('Factor Loadings Heatmap')
plt.xlabel('Factors')
plt.ylabel('Variables')
plt.show()

# Calculate and print explained variance for each factor
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues[:n_factors] / total_variance

# Append explained variance and eigenvalues to the loadings table
explained_variance_df = pd.DataFrame({
    'Explained Variance (%)': explained_variance_ratio * 100,
    'Eigenvalue': eigenvalues[:n_factors]
}, index=[f'Factor_{i+1}' for i in range(n_factors)])

print("\nExplained Variance and Eigenvalues:")
print(explained_variance_df)

# Combine loadings and explained variance for clarity
print("\nFactor Loadings:")
print(loadings_df)

# Count number of players included in the analysis who were not eliminated due to missing combine data
print(f"Number of players included in the analysis: {data.shape[0]}")

#---------------------------------------
# Archetype Analysis using Factor Scores from EFA
#---------------------------------------

# Get factor scores for each player
factor_scores = fa.transform(data_scaled)
factor_scores_df = pd.DataFrame(factor_scores, columns=[f'Factor_{i+1}' for i in range(n_factors)])

# Perform Gaussian Mixture Model Clustering on factor scores to identify archetypes 
# also determine optimal number of clusters using BIC

from sklearn.mixture import GaussianMixture
bic_scores = []
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(factor_scores_df)
    bic_scores.append(gmm.bic(factor_scores_df))
# Plot BIC scores to determine optimal number of clusters
plt.figure(figsize=(10, 5))
plt.plot(n_clusters_range, bic_scores, marker='o')
plt.title('BIC Scores for GMM Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.grid()
plt.show()

# Select optimal number of clusters based on lowest BIC score
optimal_clusters = 3
print(f"Optimal number of clusters (archetypes) based on BIC: {optimal_clusters}")  

# Fit GMM with optimal number of clusters and assign archetype labels
gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
gmm.fit(factor_scores_df)
data['Archetype'] = gmm.predict(factor_scores_df)

# Merge factor scores into data for summary stats
factor_scores_df.index = data.index
data = pd.concat([data, factor_scores_df], axis=1)

#---------------------------------------
# Visualization
#---------------------------------------

# Mean and SD for each factor score and count of players in each archetype
archetype_summary = data.groupby('Archetype')[factor_scores_df.columns].agg(['mean', 'std', 'count'])
print("\nArchetype Summary (Mean, SD, Count):")
print(archetype_summary)

# Visualize mean factor scores for each archetype using box plots
colors = ["#FF4747", "#199ACD", "#44AB59", "#FB87F5"]
plt.figure(figsize=(12, 8))
for i in range(n_factors):
    plt.subplot(1, 4, i+1)
    sns.boxplot(x='Archetype', y=f'Factor_{i+1}', data=data, hue='Archetype', palette=colors, legend=False)
    plt.title(f'Factor {i+1} Scores by Archetype')
    plt.xlabel('Archetype')
    plt.ylabel(f'Factor {i+1} Score')
plt.tight_layout()
plt.show()

# Make a grid of distribution plots for each factor by archetype (include vertical line for mean score of each archetype, keep the colors consistent with box plots, keep the line color the same as the cluster color but darker)
plt.figure(figsize=(12, 8))
for i in range(n_factors):
    plt.subplot(2, 3, i+1)
    for archetype, color in zip(data['Archetype'].unique(), colors):
        sns.kdeplot(data[data['Archetype'] == archetype][f'Factor_{i+1}'], label=f'Archetype {archetype}', fill=True, alpha=0.5, color=color)
        mean_score = data[data['Archetype'] == archetype][f'Factor_{i+1}'].mean()
        plt.axvline(mean_score, color=color, linestyle='--', linewidth=1)
    plt.title(f'Factor {i+1} Score Distribution by Archetype')
    plt.xlabel(f'Factor {i+1} Score')
plt.tight_layout()
plt.show()

#---------------------------------------
# Cluster Membership Probability Distributions
#---------------------------------------

# Get membership probabilities for each player in each cluster
cluster_probs = gmm.predict_proba(factor_scores_df)
probs_df = pd.DataFrame(cluster_probs, columns=[f'Archetype_{i}' for i in range(optimal_clusters)])
probs_df.index = data.index

# Summary statistics of membership probabilities
print("\nCluster Membership Probability Summary:")
print(probs_df.describe())

# Track max (winning) probability per player as a measure of assignment confidence
max_probs = probs_df.max(axis=1)
data['Max_Cluster_Prob'] = max_probs

plt.figure(figsize=(12, 5))

# Panel 1: distribution of max assignment probability by archetype
plt.subplot(1, 2, 1)
for archetype, color in zip(sorted(data['Archetype'].unique()), colors):
    sns.kdeplot(data[data['Archetype'] == archetype]['Max_Cluster_Prob'],
                label=f'Archetype {archetype}', fill=True, alpha=0.4, color=color)
plt.axvline(0.80, color='gray', linestyle='--', linewidth=1.2, label='0.80 threshold')
plt.title('Max Membership Probability\nDistribution by Archetype')
plt.xlabel('Max Cluster Membership Probability')
plt.ylabel('Density')
plt.legend()

# Panel 2: mean cluster probabilities per assigned archetype (stacked bar)
plt.subplot(1, 2, 2)
mean_probs_plot = probs_df.copy()
mean_probs_plot['Archetype'] = data['Archetype'].values
mean_probs_by_arch = mean_probs_plot.groupby('Archetype').mean()
mean_probs_by_arch.plot(kind='bar', stacked=True, color=colors[:optimal_clusters],
                        ax=plt.gca(), legend=True)
plt.title('Mean Cluster Membership Probabilities\nby Assigned Archetype')
plt.xlabel('Assigned Archetype')
plt.ylabel('Mean Probability')
plt.xticks(rotation=0)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Print high-confidence assignment rate and mean max probability by archetype
high_conf_pct = (max_probs >= 0.80).mean() * 100
print(f"\nPlayers with high-confidence assignment (max prob ≥ 0.80): {high_conf_pct:.1f}%")
print("\nMean max membership probability by archetype:")
print(data.groupby('Archetype')['Max_Cluster_Prob'].mean().round(3))

#---------------------------------------
# Bootstrap Stability of Cluster Assignments
#---------------------------------------

from sklearn.metrics import adjusted_rand_score

n_bootstraps = 100
bootstrap_ari = np.zeros(n_bootstraps)
np.random.seed(42)

for b in range(n_bootstraps):
    # Sample rows with replacement
    boot_idx = np.random.choice(len(factor_scores_df), size=len(factor_scores_df), replace=True)
    boot_sample = factor_scores_df.iloc[boot_idx]

    # Fit GMM on bootstrap sample, then predict labels for the full dataset
    gmm_boot = GaussianMixture(n_components=optimal_clusters, random_state=b)
    gmm_boot.fit(boot_sample)
    boot_labels = gmm_boot.predict(factor_scores_df)

    # Compare bootstrap-derived labels to original labels using ARI
    bootstrap_ari[b] = adjusted_rand_score(data['Archetype'], boot_labels)

mean_ari = bootstrap_ari.mean()
ci_lower = np.percentile(bootstrap_ari, 2.5)
ci_upper = np.percentile(bootstrap_ari, 97.5)

print(f"\nBootstrap Stability of Cluster Assignments ({n_bootstraps} iterations):")
print(f"  Mean ARI:   {mean_ari:.3f}")
print(f"  95% CI:     [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"  SD:         {bootstrap_ari.std():.3f}")

# Visualize bootstrap ARI distribution
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_ari, bins=20, color='#199ACD', edgecolor='white', alpha=0.85)
plt.axvline(mean_ari, color='#FF4747', linestyle='--', linewidth=2,
            label=f'Mean ARI = {mean_ari:.3f}')
plt.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5,
            label=f'95% CI [{ci_lower:.3f}, {ci_upper:.3f}]')
plt.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5)
plt.title('Bootstrap Stability of Cluster Assignments\n(Adjusted Rand Index)')
plt.xlabel('Adjusted Rand Index (ARI)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------------------
# Run ANOVA to determine which factors differ significantly across archetypes with bonferroni corrected pairwise t-tests for significant factors
import scipy.stats as stats
anova_results = []
for i in range(n_factors):
    groups = [data[data['Archetype'] == archetype][f'Factor_{i+1}'] for archetype in data['Archetype'].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results.append({'Factor': f'Factor_{i+1}', 'F-Statistic': f_stat, 'p-value': p_value})
anova_df = pd.DataFrame(anova_results).sort_values(by='p-value')
print("\nANOVA Results for Factor Scores by Archetype:")
print(anova_df)

# Perform Bonferroni corrected pairwise t-tests for significant factors
significant_factors = anova_df[anova_df['p-value'] < 0.05]['Factor']
for factor in significant_factors:
    print(f"\nPairwise t-tests for {factor} (Bonferroni corrected):")
    archetypes = data['Archetype'].unique()
    for i in range(len(archetypes)):
        for j in range(i+1, len(archetypes)):
            group1 = data[data['Archetype'] == archetypes[i]][factor]
            group2 = data[data['Archetype'] == archetypes[j]][factor]
            t_stat, p_value = stats.ttest_ind(group1, group2)
            p_value_corrected = p_value * (len(archetypes) * (len(archetypes) - 1) / 2)  # Bonferroni correction
            print(f"  {archetypes[i]} vs {archetypes[j]}: t-stat={t_stat:.3f}, p-value={p_value_corrected:.3f}")

#---------------------------------------
# Draft Frequency Analysis by Archetype and by year (i.e. players where DRAFT_STATUS == 'DRAFTED'), calculate the percentage of players in each year's draft (DRAFT_YEAR - 2010 and later) that belong to a specific archetype and visualize with a line plot showing trends over time for each archetype
drafted_data = data[(data['DRAFT_STATUS'] == 'DRAFTED') & (data['DRAFT_YEAR'] >= 2010)]
draft_freq = drafted_data.groupby(['DRAFT_YEAR', 'Archetype']).size().unstack(fill_value=0)
draft_freq_percent = draft_freq.div(draft_freq.sum(axis=1), axis=0) * 100
print("\nDraft Frequency by Archetype and Year (Percentage):")
print(draft_freq_percent)

# Simple linear regression for each archetype to determine if there is a significant trend in draft frequency over time
from sklearn.linear_model import LinearRegression
from scipy import stats

print("\n" + "="*80)
print("LINEAR REGRESSION: Draft Frequency Trends by Archetype Over Time")
print("="*80)

for archetype in draft_freq_percent.columns:    
    X = draft_freq_percent.index.values.reshape(-1, 1)
    y = draft_freq_percent[archetype].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions and calculate R-squared and residuals
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard error and t-statistic for p-value
    n = len(y)
    mse = ss_res / (n - 2)
    se = np.sqrt(mse / np.sum((X.flatten() - X.mean()) ** 2))
    t_stat = model.coef_[0] / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
    
    # Print regression results
    print(f"\nArchetype {int(archetype)}:")
    print(f"  Intercept (β₀):        {model.intercept_:>10.4f}")
    print(f"  Slope (β₁):            {model.coef_[0]:>10.4f}")
    print(f"  Std. Error (β₁):       {se:>10.4f}")
    print(f"  t-statistic:           {t_stat:>10.4f}")
    print(f"  p-value:               {p_value:>10.4f}")
    print(f"  R-squared:             {r_squared:>10.4f}")
    print(f"  N:                     {n:>10d}")
    
    if p_value < 0.05:
        print(f"  *** Significant trend (p < 0.05) ***")
    else:
        print(f"  (Non-significant trend)")

# Scatter plot with regression line for each archetype (using colors consistent with previous plots for each cluster), still include the regression line and confidence intervals for each archetype's trend over time, and include a legend to differentiate archetypes
plt.figure(figsize=(12, 8))
for archetype, color in zip(draft_freq_percent.columns, colors):
    sns.regplot(x=draft_freq_percent.index, y=draft_freq_percent[archetype], label=f'Archetype {int(archetype)}', color=color, scatter_kws={'s': 100}, line_kws={'linewidth': 2})
plt.title('Draft Frequency Trends by Archetype Over Time')
plt.xlabel('Draft Year')
plt.ylabel('Percentage of Drafted Players (%)')
plt.legend(title='Archetype')
plt.grid()
plt.show()

# Create a figure showing what percentage of players in each archetype were drafted. 
# include the percentage of players as a number above each bar
# also include the total number of players in each archetype as a number below each bar
archetype_counts = data['Archetype'].value_counts()
archetype_drafted_counts = data[data['DRAFT_STATUS'] == 'DRAFTED']['Archetype'].value_counts()
archetype_drafted_percent = (archetype_drafted_counts / archetype_counts) * 100
plt.figure(figsize=(10, 6))
bars = plt.bar(archetype_drafted_percent.index, archetype_drafted_percent.values, color=colors[:optimal_clusters], alpha=0.7)
plt.title('Percentage of Players Drafted by Archetype')
plt.ylabel('Percentage Drafted (%)')
plt.ylim(0, 100)
# Add percentage labels above bars and total counts below bars
for bar in bars:
    height = bar.get_height()
    archetype = int(bar.get_x() + bar.get_width() / 2)
    total_count = archetype_counts[archetype]
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.text(bar.get_x() + bar.get_width() / 2, -5, f'N={total_count}', ha='center', va='top', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#Run a statistical test to determine if the number of players drafted in each archetype is significantly different from what would be expected by chance, given the total number of players in each archetype and the overall draft rate. Use a chi-squared test for this analysis.
from scipy.stats import chi2_contingency
# Create a contingency table for drafted vs not drafted by archetype
contingency_table = pd.DataFrame({
    'Drafted': archetype_drafted_counts,
    'Not Drafted': archetype_counts - archetype_drafted_counts
}).fillna(0)
# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\nChi-Squared Test for Drafted vs Not Drafted by Archetype:")
print(f"Chi-squared statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.3f}")
if p < 0.05:    print("Result: Significant association between archetype and draft status (p < 0.05)")
else:    print("Result: No significant association between archetype and draft status (p ≥ 0.05)")

# ═══════════════════════════════════════════════════════════════
# Logistic Regression — Repeated Stratified K-Fold (500 repeats × 5 folds = 2,500 iterations)
# Two models: (A) EFA factor scores  (B) raw combine/college/anthro variables
# Primary metric: AUC-ROC. Class weights balanced to address class imbalance.
# Outputs: mean ± 95% CI for AUC and confusion-matrix-derived metrics.
# ═══════════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS   = 5
N_REPEATS  = 500

data['Drafted'] = (data['DRAFT_STATUS'] == 'DRAFTED').astype(int)
features_factor = [f'Factor_{i+1}' for i in range(n_factors)]
features_raw    = combine_variables + college_variables + anthropometric_variables

X_factor = data[features_factor].values
X_raw    = data[features_raw].values
y_lr     = data['Drafted'].values

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

def _cm_metrics(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision   = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1          = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, specificity, precision, f1, accuracy

def run_rskf(X, label):
    aucs         = []
    sensitivities = []
    specificities = []
    precisions    = []
    f1s           = []
    accuracies    = []
    # for mean ROC curve
    mean_fpr      = np.linspace(0, 1, 200)
    tprs          = []

    for train_idx, test_idx in rskf.split(X, y_lr):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_lr[train_idx], y_lr[test_idx]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_tr)

        proba  = clf.predict_proba(X_te)[:, 1]
        preds  = clf.predict(X_te)

        auc = roc_auc_score(y_te, proba)
        aucs.append(auc)

        fpr_i, tpr_i, _ = roc_curve(y_te, proba)
        tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))

        tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
        sens, spec, prec, f1, acc = _cm_metrics(tn, fp, fn, tp)
        sensitivities.append(sens)
        specificities.append(spec)
        precisions.append(prec)
        f1s.append(f1)
        accuracies.append(acc)

    total = N_SPLITS * N_REPEATS
    print(f"\n{'═'*60}")
    print(f"{label} — Repeated Stratified K-Fold ({N_REPEATS} repeats × {N_SPLITS} folds = {total} iterations)")
    print(f"{'═'*60}")

    def ci_str(arr, decimals=3):
        arr = np.array(arr)
        mu  = np.nanmean(arr)
        lo  = np.nanpercentile(arr, 2.5)
        hi  = np.nanpercentile(arr, 97.5)
        fmt = f'.{decimals}f'
        return f"{mu:{fmt}}  (95% CI: {lo:{fmt}} – {hi:{fmt}})"

    print(f"  AUC          : {ci_str(aucs)}")
    print(f"  Accuracy     : {ci_str(accuracies)}")
    print(f"  Sensitivity  : {ci_str(sensitivities)}")
    print(f"  Specificity  : {ci_str(specificities)}")
    print(f"  Precision    : {ci_str(precisions)}")
    print(f"  F1 Score     : {ci_str(f1s)}")

    # Mean ROC curve ± 95% CI band
    mean_tpr  = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    lo_tpr    = np.percentile(tprs, 2.5,  axis=0)
    hi_tpr    = np.percentile(tprs, 97.5, axis=0)
    mean_auc  = np.mean(aucs)
    auc_lo    = np.percentile(aucs, 2.5)
    auc_hi    = np.percentile(aucs, 97.5)

    return dict(
        label=label,
        aucs=np.array(aucs),
        mean_auc=mean_auc, auc_lo=auc_lo, auc_hi=auc_hi,
        mean_fpr=mean_fpr, mean_tpr=mean_tpr,
        lo_tpr=lo_tpr, hi_tpr=hi_tpr,
        sensitivities=sensitivities, specificities=specificities,
        precisions=precisions, f1s=f1s, accuracies=accuracies
    )

print(f"\nRunning {N_SPLITS * N_REPEATS} iterations per model — this may take a moment...")
res_factor = run_rskf(X_factor, 'Model A: Factor Scores')
res_raw    = run_rskf(X_raw,    'Model B: Raw Variables')

# ── ROC curves (mean ± 95% CI band, both models on one plot) ──
plt.figure(figsize=(8, 6))
for res, color in [(res_factor, '#199ACD'), (res_raw, '#FF4747')]:
    lbl = (f"{res['label']}\n"
           f"AUC = {res['mean_auc']:.3f} (95% CI: {res['auc_lo']:.3f} – {res['auc_hi']:.3f})")
    plt.plot(res['mean_fpr'], res['mean_tpr'], color=color, linewidth=2, label=lbl)
    plt.fill_between(res['mean_fpr'], res['lo_tpr'], res['hi_tpr'],
                     color=color, alpha=0.15)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Mean ROC Curves ± 95% CI\n({N_REPEATS} repeats × {N_SPLITS}-fold = {N_SPLITS * N_REPEATS} iterations)')
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── AUC distribution comparison ──
plt.figure(figsize=(9, 5))
sns.histplot(res_factor['aucs'], bins=60, kde=True, color='#199ACD', alpha=0.5,
             label=f"Factor Scores (μ={res_factor['mean_auc']:.3f})")
sns.histplot(res_raw['aucs'],    bins=60, kde=True, color='#FF4747',  alpha=0.5,
             label=f"Raw Variables (μ={res_raw['mean_auc']:.3f})")
plt.axvline(res_factor['mean_auc'], color='#199ACD', linestyle='--', linewidth=1.5)
plt.axvline(res_raw['mean_auc'],    color='#FF4747',  linestyle='--', linewidth=1.5)
plt.xlabel('AUC')
plt.title(f'AUC Distribution — {N_SPLITS * N_REPEATS} Iterations')
plt.legend()
plt.tight_layout()
plt.show()

# ── Confusion-matrix metric summary bar chart (mean ± 95% CI) ──
metrics_labels = ['Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'Accuracy']
factor_vals = [np.nanmean(res_factor['sensitivities']), np.nanmean(res_factor['specificities']),
               np.nanmean(res_factor['precisions']),    np.nanmean(res_factor['f1s']),
               np.nanmean(res_factor['accuracies'])]
raw_vals    = [np.nanmean(res_raw['sensitivities']), np.nanmean(res_raw['specificities']),
               np.nanmean(res_raw['precisions']),    np.nanmean(res_raw['f1s']),
               np.nanmean(res_raw['accuracies'])]
factor_lo   = [np.nanpercentile(res_factor['sensitivities'], 2.5), np.nanpercentile(res_factor['specificities'], 2.5),
               np.nanpercentile(res_factor['precisions'], 2.5),    np.nanpercentile(res_factor['f1s'], 2.5),
               np.nanpercentile(res_factor['accuracies'], 2.5)]
factor_hi   = [np.nanpercentile(res_factor['sensitivities'], 97.5), np.nanpercentile(res_factor['specificities'], 97.5),
               np.nanpercentile(res_factor['precisions'], 97.5),    np.nanpercentile(res_factor['f1s'], 97.5),
               np.nanpercentile(res_factor['accuracies'], 97.5)]
raw_lo      = [np.nanpercentile(res_raw['sensitivities'], 2.5), np.nanpercentile(res_raw['specificities'], 2.5),
               np.nanpercentile(res_raw['precisions'], 2.5),    np.nanpercentile(res_raw['f1s'], 2.5),
               np.nanpercentile(res_raw['accuracies'], 2.5)]
raw_hi      = [np.nanpercentile(res_raw['sensitivities'], 97.5), np.nanpercentile(res_raw['specificities'], 97.5),
               np.nanpercentile(res_raw['precisions'], 97.5),    np.nanpercentile(res_raw['f1s'], 97.5),
               np.nanpercentile(res_raw['accuracies'], 97.5)]

x = np.arange(len(metrics_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, factor_vals,
       yerr=[np.subtract(factor_vals, factor_lo), np.subtract(factor_hi, factor_vals)],
       width=width, label='Factor Scores', color='#199ACD', ecolor='black', capsize=5,
       edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, raw_vals,
       yerr=[np.subtract(raw_vals, raw_lo), np.subtract(raw_hi, raw_vals)],
       width=width, label='Raw Variables', color='#FF4747', ecolor='black', capsize=5,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score (Mean ± 95% CI)')
ax.set_title(f'Classification Metrics — Mean ± 95% CI\n({N_SPLITS * N_REPEATS} Repeated Stratified K-Fold iterations)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# give the model coefficients for the final model fit on the dataset so we can determine which factors had a significant influence on predictions. include beta coefficients and p values
# Fit final logistic regression model on entire dataset using factor scores
sc_final = StandardScaler()
X_final = sc_final.fit_transform(X_factor)
y_final = y_lr
final_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
final_model.fit(X_final, y_final)   
# Get coefficients and calculate p-values using Wald test
coefficients = final_model.coef_[0]
intercept = final_model.intercept_[0]
# Calculate standard errors and p-values
preds = final_model.predict_proba(X_final)[:, 1]
X_design = np.hstack([np.ones((X_final.shape[0], 1)), X_final])
var_covar_matrix = np.linalg.inv(X_design.T @ np.diag(preds * (1 - preds)) @ X_design)
standard_errors = np.sqrt(np.diag(var_covar_matrix))
z_scores = np.hstack([intercept, coefficients]) / standard_errors
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
# Create a summary table of coefficients and p-values
coef_summary = pd.DataFrame({
    'Variable': ['Intercept'] + features_factor,
    'Coefficient': np.hstack([intercept, coefficients]),
    'Std. Error': standard_errors,
    'z-score': z_scores,
    'p-value': p_values
})
print("\nFinal Logistic Regression Model Coefficients:")
print(coef_summary)