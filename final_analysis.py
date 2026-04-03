import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, wilcoxon, kruskal, mannwhitneyu
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, ElasticNetCV
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             mean_squared_error, r2_score, adjusted_rand_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from factor_analyzer import FactorAnalyzer
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

data = pd.read_csv('/Users/bendoyle/Library/CloudStorage/OneDrive-TexasA&MUniversity/'
                   'BD-JDT-TAMU-Sport-Data-Challenge-2026/Post-Query Datasets/PRE_NFL_DATA.csv')

college_variables = ['COLLEGE_SACKS', 'COLLEGE_FORCED_FUMBLES',
                     'COLLEGE_TACKLES_SOLO', 'COLLEGE_TACKLES_ASSISTED',
                     'COLLEGE_PASSES_DEFENDED', 'COLLEGE_INTERCEPTIONS']
anthropometric_variables = ['HEIGHT_IN', 'WEIGHT_LB']

print(f"Total players before filtering: {data.shape[0]}")
data = data[data['COMBINE_TESTS_MISSING'] <= 3].copy()
print(f"Players after filtering (≤3 missing combine tests): {data.shape[0]}")

data['X40Y_VEL'] = 36.57 / data['X40Y_SEC']
data['X3CD_VEL'] = 27.43 / data['X3CD_SEC']
data['PROA_VEL'] = 18.28 / data['PROA_SEC']
data['mCODD']    = data['PROA_SEC'] / data['X40Y_SEC']

combine_variables = ['VJ_IN', 'SLJ_IN', 'BENCH_REPS', 'P40',
                     'MANN_SLJP', 'X40Y_VEL', 'X3CD_VEL', 'PROA_VEL']

def random_forest_imputation(df, target_col, feature_cols):
    known   = df[df[target_col].notnull()]
    unknown = df[df[target_col].isnull()]
    if unknown.empty:
        return df
    imputer     = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(known[feature_cols])
    X_unk_imp   = imputer.transform(unknown[feature_cols])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imp, known[target_col])
    df.loc[df[target_col].isnull(), target_col] = model.predict(X_unk_imp)
    return df

for var in combine_variables:
    data = random_forest_imputation(
        data, var,
        [c for c in combine_variables if c != var] + anthropometric_variables
    )

fa_vars     = combine_variables + college_variables + anthropometric_variables
scaler      = StandardScaler()
data_scaled = scaler.fit_transform(data[fa_vars])

raw_col_names      = [f'RAW_{v}' for v in fa_vars]
data[raw_col_names] = data_scaled
raw_variables      = raw_col_names

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLORATORY FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

fa_screen = FactorAnalyzer(rotation=None)
fa_screen.fit(data_scaled)
eigenvalues, _ = fa_screen.get_eigenvalues()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.grid()
plt.tight_layout()
plt.show()

n_factors = 4
fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
fa.fit(data_scaled)

loadings_df = pd.DataFrame(fa.loadings_, index=fa_vars,
                           columns=[f'Factor_{i+1}' for i in range(n_factors)])
print("\nFactor Loadings:")
print(loadings_df.round(3))

total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues[:n_factors] / total_variance
explained_variance_df = pd.DataFrame({
    'Explained Variance (%)': (explained_variance_ratio * 100).round(2),
    'Eigenvalue':             eigenvalues[:n_factors].round(4)
}, index=[f'Factor_{i+1}' for i in range(n_factors)])
print("\nExplained Variance and Eigenvalues:")
print(explained_variance_df)

plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
plt.title('Factor Loadings Heatmap')
plt.xlabel('Factors')
plt.ylabel('Variables')
plt.tight_layout()
plt.show()

print(f"Number of players included in the analysis: {data.shape[0]}")

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE ANALYSIS (GMM on Factor Scores)
# ═══════════════════════════════════════════════════════════════════════════════

factor_scores    = fa.transform(data_scaled)
factor_scores_df = pd.DataFrame(factor_scores, index=data.index,
                                columns=[f'Factor_{i+1}' for i in range(n_factors)])
factor_variables = list(factor_scores_df.columns)

bic_scores = []
n_clusters_range = range(1, 10)
for n_clusters in n_clusters_range:
    gmm_tmp = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_tmp.fit(factor_scores_df)
    bic_scores.append(gmm_tmp.bic(factor_scores_df))

plt.figure(figsize=(10, 5))
plt.plot(n_clusters_range, bic_scores, marker='o')
plt.title('BIC Scores for GMM Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.grid()
plt.show()

optimal_clusters = 3
print(f"Optimal number of clusters (archetypes) based on BIC: {optimal_clusters}")

gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
gmm.fit(factor_scores_df)
data['Archetype'] = gmm.predict(factor_scores_df)
data = pd.concat([data, factor_scores_df], axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

colors = ["#FF4747", "#199ACD", "#44AB59", "#FB87F5"]

archetype_summary = data.groupby('Archetype')[factor_variables].agg(['mean', 'std', 'count'])
print("\nArchetype Summary (Mean, SD, Count):")
print(archetype_summary)

plt.figure(figsize=(12, 8))
for i in range(n_factors):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(x='Archetype', y=f'Factor_{i+1}', data=data,
                hue='Archetype', palette=colors, legend=False)
    plt.title(f'Factor {i+1} Scores by Archetype')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i in range(n_factors):
    plt.subplot(2, 3, i + 1)
    for archetype, color in zip(data['Archetype'].unique(), colors):
        subset = data[data['Archetype'] == archetype][f'Factor_{i+1}']
        sns.kdeplot(subset, label=f'Archetype {archetype}', fill=True, alpha=0.5, color=color)
        plt.axvline(subset.mean(), color=color, linestyle='--', linewidth=1)
    plt.title(f'Factor {i+1} Score Distribution by Archetype')
plt.tight_layout()
plt.show()

# ════════════════════════════════════════���══════════════════════════════════════
# CLUSTER MEMBERSHIP PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════════════

cluster_probs = gmm.predict_proba(factor_scores_df)
probs_df = pd.DataFrame(cluster_probs,
                        columns=[f'Archetype_{i}' for i in range(optimal_clusters)],
                        index=data.index)

print("\nCluster Membership Probability Summary:")
print(probs_df.describe())

max_probs = probs_df.max(axis=1)
data['Max_Cluster_Prob'] = max_probs

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for archetype, color in zip(sorted(data['Archetype'].unique()), colors):
    sns.kdeplot(data[data['Archetype'] == archetype]['Max_Cluster_Prob'],
                label=f'Archetype {archetype}', fill=True, alpha=0.4, color=color)
plt.axvline(0.80, color='gray', linestyle='--', linewidth=1.2, label='0.80 threshold')
plt.title('Max Membership Probability\nDistribution by Archetype')
plt.xlabel('Max Cluster Membership Probability')
plt.ylabel('Density')
plt.legend()

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

high_conf_pct = (max_probs >= 0.80).mean() * 100
print(f"\nPlayers with high-confidence assignment (max prob ≥ 0.80): {high_conf_pct:.1f}%")
print("\nMean max membership probability by archetype:")
print(data.groupby('Archetype')['Max_Cluster_Prob'].mean().round(3))

# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

n_bootstraps   = 100
bootstrap_ari  = np.zeros(n_bootstraps)
np.random.seed(42)

for b in range(n_bootstraps):
    boot_idx    = np.random.choice(len(factor_scores_df), size=len(factor_scores_df), replace=True)
    boot_sample = factor_scores_df.iloc[boot_idx]
    gmm_boot    = GaussianMixture(n_components=optimal_clusters, random_state=b)
    gmm_boot.fit(boot_sample)
    boot_labels = gmm_boot.predict(factor_scores_df)
    bootstrap_ari[b] = adjusted_rand_score(data['Archetype'], boot_labels)

mean_ari = bootstrap_ari.mean()
ci_lower = np.percentile(bootstrap_ari, 2.5)
ci_upper = np.percentile(bootstrap_ari, 97.5)

print(f"\nBootstrap Stability ({n_bootstraps} iterations):")
print(f"  Mean ARI: {mean_ari:.3f}  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]  SD: {bootstrap_ari.std():.3f}")

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

# ═══════════════════════════════════════════════════════════════════════════════
# ANOVA + BONFERRONI PAIRWISE T-TESTS
# ═══════════════════════════════════════════════════════════════════════════════

anova_results = []
for i in range(n_factors):
    groups = [data[data['Archetype'] == a][f'Factor_{i+1}'] for a in data['Archetype'].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results.append({'Factor': f'Factor_{i+1}', 'F-Statistic': f_stat, 'p-value': p_value})
anova_df = pd.DataFrame(anova_results).sort_values(by='p-value')
print("\nANOVA Results for Factor Scores by Archetype:")
print(anova_df)

significant_factors = anova_df[anova_df['p-value'] < 0.05]['Factor']
for factor in significant_factors:
    print(f"\nPairwise t-tests for {factor} (Bonferroni corrected):")
    archetypes = data['Archetype'].unique()
    for i in range(len(archetypes)):
        for j in range(i + 1, len(archetypes)):
            g1 = data[data['Archetype'] == archetypes[i]][factor]
            g2 = data[data['Archetype'] == archetypes[j]][factor]
            t_stat, p_val = stats.ttest_ind(g1, g2)
            p_corr = p_val * (len(archetypes) * (len(archetypes) - 1) / 2)
            print(f"  {archetypes[i]} vs {archetypes[j]}: t={t_stat:.3f}, p={p_corr:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT FREQUENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

drafted_data = data[(data['DRAFT_STATUS'] == 'DRAFTED') & (data['DRAFT_YEAR'] >= 2010)]
draft_freq = drafted_data.groupby(['DRAFT_YEAR', 'Archetype']).size().unstack(fill_value=0)
draft_freq_percent = draft_freq.div(draft_freq.sum(axis=1), axis=0) * 100
print("\nDraft Frequency by Archetype and Year (Percentage):")
print(draft_freq_percent)

print("\n" + "=" * 80)
print("LINEAR REGRESSION: Draft Frequency Trends by Archetype Over Time")
print("=" * 80)

for archetype in draft_freq_percent.columns:
    X_reg = draft_freq_percent.index.values.reshape(-1, 1)
    y_reg = draft_freq_percent[archetype].values
    lr_model = LinearRegression().fit(X_reg, y_reg)
    y_pred = lr_model.predict(X_reg)
    ss_res = np.sum((y_reg - y_pred) ** 2)
    ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
    r_sq   = 1 - (ss_res / ss_tot)
    n_obs  = len(y_reg)
    mse    = ss_res / (n_obs - 2)
    se     = np.sqrt(mse / np.sum((X_reg.flatten() - X_reg.mean()) ** 2))
    t_s    = lr_model.coef_[0] / se
    p_v    = 2 * (1 - stats.t.cdf(np.abs(t_s), n_obs - 2))
    print(f"\nArchetype {int(archetype)}:  β₁={lr_model.coef_[0]:.4f}  SE={se:.4f}  "
          f"t={t_s:.4f}  p={p_v:.4f}  R²={r_sq:.4f}  N={n_obs}"
          f"  {'*** Sig' if p_v < 0.05 else '(ns)'}")

plt.figure(figsize=(12, 8))
for archetype, color in zip(draft_freq_percent.columns, colors):
    sns.regplot(x=draft_freq_percent.index, y=draft_freq_percent[archetype],
                label=f'Archetype {int(archetype)}', color=color,
                scatter_kws={'s': 100}, line_kws={'linewidth': 2})
plt.title('Draft Frequency Trends by Archetype Over Time')
plt.xlabel('Draft Year')
plt.ylabel('Percentage of Drafted Players (%)')
plt.legend(title='Archetype')
plt.grid()
plt.show()

archetype_counts         = data['Archetype'].value_counts()
archetype_drafted_counts = data[data['DRAFT_STATUS'] == 'DRAFTED']['Archetype'].value_counts()
archetype_drafted_percent = (archetype_drafted_counts / archetype_counts) * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(archetype_drafted_percent.index, archetype_drafted_percent.values,
               color=colors[:optimal_clusters], alpha=0.7)
plt.title('Percentage of Players Drafted by Archetype')
plt.ylabel('Percentage Drafted (%)')
plt.ylim(0, 100)
for bar in bars:
    height    = bar.get_height()
    archetype = int(bar.get_x() + bar.get_width() / 2)
    total     = archetype_counts[archetype]
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.text(bar.get_x() + bar.get_width() / 2, -5,
             f'N={total}', ha='center', va='top', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

contingency_table = pd.DataFrame({
    'Drafted':     archetype_drafted_counts,
    'Not Drafted': archetype_counts - archetype_drafted_counts
}).fillna(0)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Squared Test: χ²={chi2:.3f}  df={dof}  p={p:.3f}")
if p < 0.05:
    print("Result: Significant association between archetype and draft status (p < 0.05)")
else:
    print("Result: No significant association (p ≥ 0.05)")

# ═════════════════════════════════��═════════════════════════════════════════════
# LOGISTIC REGRESSION — DRAFT PREDICTION (Factor Scores vs Raw Variables)
# ═══════════════════════════════════════════════════════════════════════════════

N_SPLITS  = 5
N_REPEATS = 500

data['Drafted']    = (data['DRAFT_STATUS'] == 'DRAFTED').astype(int)
features_factor_lr = factor_variables
features_raw_lr    = combine_variables + college_variables + anthropometric_variables

X_factor_lr = data[features_factor_lr].values
X_raw_lr    = data[features_raw_lr].values
y_lr        = data['Drafted'].values

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

def _cm_metrics(tn, fp, fn, tp):
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1   = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return sens, spec, prec, f1, acc

def run_rskf(X, label):
    aucs, sensitivities, specificities = [], [], []
    precisions, f1s, accuracies = [], [], []
    mean_fpr = np.linspace(0, 1, 200)
    tprs     = []
    for train_idx, test_idx in rskf.split(X, y_lr):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_lr[train_idx], y_lr[test_idx]
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf  = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
        auc   = roc_auc_score(y_te, proba)
        aucs.append(auc)
        fpr_i, tpr_i, _ = roc_curve(y_te, proba)
        tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))
        tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
        s, sp, pr, f, ac = _cm_metrics(tn, fp, fn, tp)
        sensitivities.append(s); specificities.append(sp)
        precisions.append(pr); f1s.append(f); accuracies.append(ac)

    total = N_SPLITS * N_REPEATS
    print(f"\n{'═'*60}")
    print(f"{label} — {N_REPEATS}×{N_SPLITS}-fold = {total} iterations")
    print(f"{'═'*60}")
    def ci_str(arr, d=3):
        a = np.array(arr)
        return (f"{np.nanmean(a):.{d}f}  "
                f"(95% CI: {np.nanpercentile(a,2.5):.{d}f} – {np.nanpercentile(a,97.5):.{d}f})")
    print(f"  AUC         : {ci_str(aucs)}")
    print(f"  Accuracy    : {ci_str(accuracies)}")
    print(f"  Sensitivity : {ci_str(sensitivities)}")
    print(f"  Specificity : {ci_str(specificities)}")
    print(f"  Precision   : {ci_str(precisions)}")
    print(f"  F1 Score    : {ci_str(f1s)}")

    mean_tpr     = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    lo_tpr       = np.percentile(tprs, 2.5, axis=0)
    hi_tpr       = np.percentile(tprs, 97.5, axis=0)
    mean_auc     = np.mean(aucs)
    auc_lo       = np.percentile(aucs, 2.5)
    auc_hi       = np.percentile(aucs, 97.5)
    return dict(label=label, aucs=np.array(aucs),
                mean_auc=mean_auc, auc_lo=auc_lo, auc_hi=auc_hi,
                mean_fpr=mean_fpr, mean_tpr=mean_tpr,
                lo_tpr=lo_tpr, hi_tpr=hi_tpr,
                sensitivities=sensitivities, specificities=specificities,
                precisions=precisions, f1s=f1s, accuracies=accuracies)

print(f"\nRunning {N_SPLITS * N_REPEATS} iterations per model...")
res_factor_lr = run_rskf(X_factor_lr, 'Model A: Factor Scores')
res_raw_lr    = run_rskf(X_raw_lr,    'Model B: Raw Variables')

plt.figure(figsize=(8, 6))
for res, color in [(res_factor_lr, '#199ACD'), (res_raw_lr, '#FF4747')]:
    lbl = (f"{res['label']}\n"
           f"AUC = {res['mean_auc']:.3f} (95% CI: {res['auc_lo']:.3f} – {res['auc_hi']:.3f})")
    plt.plot(res['mean_fpr'], res['mean_tpr'], color=color, linewidth=2, label=lbl)
    plt.fill_between(res['mean_fpr'], res['lo_tpr'], res['hi_tpr'], color=color, alpha=0.15)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Mean ROC Curves ± 95% CI\n({N_REPEATS}×{N_SPLITS}-fold = {N_SPLITS*N_REPEATS} iterations)')
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.histplot(res_factor_lr['aucs'], bins=60, kde=True, color='#199ACD', alpha=0.5,
             label=f"Factor Scores (μ={res_factor_lr['mean_auc']:.3f})")
sns.histplot(res_raw_lr['aucs'], bins=60, kde=True, color='#FF4747', alpha=0.5,
             label=f"Raw Variables (μ={res_raw_lr['mean_auc']:.3f})")
plt.axvline(res_factor_lr['mean_auc'], color='#199ACD', linestyle='--', linewidth=1.5)
plt.axvline(res_raw_lr['mean_auc'],    color='#FF4747',  linestyle='--', linewidth=1.5)
plt.xlabel('AUC')
plt.title(f'AUC Distribution — {N_SPLITS*N_REPEATS} Iterations')
plt.legend()
plt.tight_layout()
plt.show()

metrics_labels = ['Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'Accuracy']
factor_vals = [np.nanmean(res_factor_lr[k]) for k in
               ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]
raw_vals    = [np.nanmean(res_raw_lr[k]) for k in
               ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]
factor_lo = [np.nanpercentile(res_factor_lr[k], 2.5) for k in
             ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]
factor_hi = [np.nanpercentile(res_factor_lr[k], 97.5) for k in
             ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]
raw_lo    = [np.nanpercentile(res_raw_lr[k], 2.5) for k in
             ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]
raw_hi    = [np.nanpercentile(res_raw_lr[k], 97.5) for k in
             ['sensitivities', 'specificities', 'precisions', 'f1s', 'accuracies']]

x     = np.arange(len(metrics_labels))
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
ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score (Mean ± 95% CI)')
ax.set_title(f'Classification Metrics — Mean ± 95% CI\n({N_SPLITS*N_REPEATS} iterations)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

sc_final    = StandardScaler()
X_final     = sc_final.fit_transform(X_factor_lr)
final_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
final_model.fit(X_final, y_lr)
coefficients = final_model.coef_[0]
intercept    = final_model.intercept_[0]
preds_prob   = final_model.predict_proba(X_final)[:, 1]
X_design     = np.hstack([np.ones((X_final.shape[0], 1)), X_final])
V            = np.linalg.inv(X_design.T @ np.diag(preds_prob * (1 - preds_prob)) @ X_design)
se           = np.sqrt(np.diag(V))
z            = np.hstack([intercept, coefficients]) / se
p_vals       = 2 * (1 - stats.norm.cdf(np.abs(z)))
coef_summary = pd.DataFrame({
    'Variable':    ['Intercept'] + features_factor_lr,
    'Coefficient': np.hstack([intercept, coefficients]),
    'Std. Error':  se, 'z-score': z, 'p-value': p_vals
})
print("\nFinal Logistic Regression Coefficients:")
print(coef_summary)

# ═══════════════════════════════════════════════════════════════════════════════
# NFL OUTCOME ANALYSIS — ElasticNet Models
# ═══════════════════════════════════════════════════════════════════════════════

nfl = pd.read_csv('/Users/bendoyle/Library/CloudStorage/OneDrive-TexasA&MUniversity/'
                  'BD-JDT-TAMU-Sport-Data-Challenge-2026/Post-Query Datasets/nfl_data.csv')

nfl = nfl[nfl['NFL_CAREER_GAMES_PLAYED'] >= 10].copy()
print(f"\nNFL players with ≥10 games: {nfl.shape[0]}")

impact_raw   = nfl['NFL_CAREER_SACKS'].fillna(0) + nfl['NFL_CAREER_TFL'].fillna(0)
impact_log   = impact_raw.apply(lambda x: np.log1p(x) if x > 0 else 0)
impact_score = (impact_log - impact_log.mean()) / impact_log.std()
nfl['IMPACT_SCORE'] = impact_score

join_keys  = ['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR']
model_data = (nfl[join_keys + ['IMPACT_SCORE']]
              .merge(data[join_keys + factor_variables + raw_variables],
                     on=join_keys, how='inner'))

print(f"Modeling dataset: {model_data.shape[0]} players")

plt.figure(figsize=(8, 5))
sns.histplot(model_data['IMPACT_SCORE'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Impact Score (log1p, z-scored)')
plt.xlabel('Impact Score')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

y_nfl      = model_data['IMPACT_SCORE']
l1_ratios  = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
alpha_grid = np.logspace(-4, 0, 50)
N_ITER     = 1000

def run_enet_model(X, feature_names, label, color):
    en_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alpha_grid,
                         cv=5, max_iter=10000, random_state=42, n_jobs=-1)
    en_cv.fit(X, y_nfl)
    best_alpha    = en_cv.alpha_
    best_l1_ratio = en_cv.l1_ratio_
    print(f"\n{label} — ElasticNetCV: α={best_alpha:.6f}  L1={best_l1_ratio:.2f}")

    mc_coefs = np.zeros((N_ITER, len(feature_names)))
    mc_rmse  = np.zeros(N_ITER)
    mc_r2    = np.zeros(N_ITER)
    print(f"Running {N_ITER} Monte Carlo iterations ({label})...")
    for i in range(N_ITER):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_nfl, test_size=0.2, random_state=i)
        en = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                        max_iter=10000, random_state=i)
        en.fit(X_tr, y_tr)
        y_p          = en.predict(X_te)
        mc_coefs[i]  = en.coef_
        mc_rmse[i]   = np.sqrt(mean_squared_error(y_te, y_p))
        mc_r2[i]     = r2_score(y_te, y_p)
    print("Done.")

    mc_coef_df = pd.DataFrame({
        'Feature':      feature_names,
        'Mean Coef':    mc_coefs.mean(axis=0).round(4),
        'SD':           mc_coefs.std(axis=0).round(4),
        '95% CI Lower': np.percentile(mc_coefs, 2.5, axis=0).round(4),
        '95% CI Upper': np.percentile(mc_coefs, 97.5, axis=0).round(4),
        '% Non-zero':   ((mc_coefs != 0).mean(axis=0) * 100).round(1)
    }).sort_values('Mean Coef', key=abs, ascending=False).reset_index(drop=True)

    print(f"\n── {label}: MC Coefficient Summary ──")
    print(mc_coef_df.to_string(index=False))
    print(f"\n── {label}: Performance ──")
    for mname, vals in [('RMSE', mc_rmse), ('R²', mc_r2)]:
        print(f"  {mname}: Mean={vals.mean():.4f}  SD={vals.std():.4f}  "
              f"95% CI [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_nfl, test_size=0.2, random_state=42)
    en_final = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                          max_iter=10000, random_state=42)
    en_final.fit(X_tr, y_tr)
    y_pred_f = en_final.predict(X_te)
    rmse_f   = np.sqrt(mean_squared_error(y_te, y_pred_f))
    r2_f     = r2_score(y_te, y_pred_f)
    print(f"\n{label} final (seed=42) — RMSE: {rmse_f:.4f} | R²: {r2_f:.4f}")

    perm_res = permutation_importance(en_final, X_te, y_te,
                                      n_repeats=30, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        'Feature':         feature_names,
        'Importance Mean': perm_res.importances_mean,
        'Importance Std':  perm_res.importances_std
    }).sort_values('Importance Mean', ascending=False).reset_index(drop=True)
    print(f"\n{label} — Permutation Importances:")
    print(perm_df.to_string(index=False))

    return dict(label=label, color=color,
                mc_r2=mc_r2, mc_rmse=mc_rmse,
                en_final=en_final, best_alpha=best_alpha, best_l1_ratio=best_l1_ratio,
                X_test=X_te, y_test=y_te, y_pred=y_pred_f,
                rmse=rmse_f, r2=r2_f,
                perm_df=perm_df, mc_coefs=mc_coefs, mc_coef_df=mc_coef_df,
                feature_names=feature_names)

def run_rfecv(X_all, variables, color, title_tag):
    rfecv = RFECV(estimator=Ridge(alpha=1.0), step=1, cv=5, scoring='r2', n_jobs=-1)
    rfecv.fit(X_all, y_nfl)
    selected   = [v for v, s in zip(variables, rfecv.support_) if s]
    eliminated = [v for v, s in zip(variables, rfecv.support_) if not s]
    print(f"\nRFECV ({title_tag}): {rfecv.n_features_}/{len(variables)} selected")
    print(f"  Selected  : {selected}")
    print(f"  Eliminated: {eliminated}")
    n_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
    mu  = rfecv.cv_results_['mean_test_score']
    sig = rfecv.cv_results_['std_test_score']
    plt.figure(figsize=(9, 4))
    plt.plot(n_range, mu, marker='o', color=color)
    plt.fill_between(n_range, mu - sig, mu + sig, alpha=0.25, color=color)
    plt.axvline(rfecv.n_features_, color='red', linestyle='--',
                label=f'Optimal = {rfecv.n_features_}')
    plt.xlabel('Number of Features'); plt.ylabel('CV R²')
    plt.title(f'RFECV — {title_tag}'); plt.legend(); plt.grid()
    plt.tight_layout(); plt.show()
    return selected

print("\n" + "═" * 60)
print("MODEL A — EFA Factors (standard ElasticNet)")
print("═" * 60)
res_a = run_enet_model(model_data[factor_variables], factor_variables,
                       'Model A (Factors)', 'steelblue')

print("\n" + "═" * 60)
print("MODEL B — EFA Factors + RFECV")
print("═" * 60)
sel_factors = run_rfecv(model_data[factor_variables], factor_variables,
                        'steelblue', 'Model B (Factors + RFECV)')
res_b = run_enet_model(model_data[sel_factors], sel_factors,
                       'Model B (Factors + RFECV)', '#1A6FA8')

print("\n" + "═" * 60)
print("MODEL C — Raw Features (standard ElasticNet)")
print("═" * 60)
res_c = run_enet_model(model_data[raw_variables], raw_variables,
                       'Model C (Raw)', 'darkorange')

print("\n" + "═" * 60)
print("MODEL D — Raw Features + RFECV")
print("═" * 60)
sel_raw = run_rfecv(model_data[raw_variables], raw_variables,
                    'darkorange', 'Model D (Raw + RFECV)')
res_d = run_enet_model(model_data[sel_raw], sel_raw,
                       'Model D (Raw + RFECV)', '#B85C00')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

all_results = [res_a, res_b, res_c, res_d]

comp_df = pd.DataFrame({
    'Model':           [r['label'] for r in all_results],
    'N Features':      [len(r['feature_names']) for r in all_results],
    'Mean R²':         [round(r['mc_r2'].mean(), 4) for r in all_results],
    'SD R²':           [round(r['mc_r2'].std(), 4) for r in all_results],
    'R² CI Lower':     [round(np.percentile(r['mc_r2'], 2.5), 4) for r in all_results],
    'R² CI Upper':     [round(np.percentile(r['mc_r2'], 97.5), 4) for r in all_results],
    'Mean RMSE':       [round(r['mc_rmse'].mean(), 4) for r in all_results],
    'Final R² (s=42)': [round(r['r2'], 4) for r in all_results],
    'Final RMSE(s=42)':[round(r['rmse'], 4) for r in all_results],
})
print("\n" + "═" * 60)
print("MODEL COMPARISON")
print("═" * 60)
print(comp_df.to_string(index=False))

plt.figure(figsize=(12, 5))
for r in all_results:
    sns.histplot(r['mc_r2'], bins=40, kde=True, alpha=0.35, color=r['color'],
                 label=f"{r['label']} (μ={r['mc_r2'].mean():.3f})")
    plt.axvline(r['mc_r2'].mean(), color=r['color'], linestyle='--', linewidth=1.5)
plt.xlabel('R²')
plt.title(f'R² Distribution — All Models ({N_ITER} MC iterations)')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 5))
for r in all_results:
    sns.histplot(r['mc_rmse'], bins=40, kde=True, alpha=0.35, color=r['color'],
                 label=f"{r['label']} (μ={r['mc_rmse'].mean():.3f})")
    plt.axvline(r['mc_rmse'].mean(), color=r['color'], linestyle='--', linewidth=1.5)
plt.xlabel('RMSE')
plt.title(f'RMSE Distribution — All Models ({N_ITER} MC iterations)')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
x_bar  = np.arange(len(all_results))
means  = [r['mc_r2'].mean() for r in all_results]
lo_bar = [r['mc_r2'].mean() - np.percentile(r['mc_r2'], 2.5) for r in all_results]
hi_bar = [np.percentile(r['mc_r2'], 97.5) - r['mc_r2'].mean() for r in all_results]
ax.bar(x_bar, means, yerr=[lo_bar, hi_bar],
       color=[r['color'] for r in all_results],
       edgecolor='black', linewidth=0.6, capsize=6, width=0.55)
ax.set_xticks(x_bar)
ax.set_xticklabels([r['label'] for r in all_results], rotation=15, ha='right')
ax.set_ylabel('Mean R² (95% CI)')
ax.set_title(f'Model Comparison — Mean R² ± 95% CI ({N_ITER} iterations)')
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout(); plt.show()

# Predicted vs Actual
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, r in zip(axes.flat, all_results):
    y_t, y_p = r['y_test'], r['y_pred']
    ax.scatter(y_t, y_p, alpha=0.6, edgecolors='black', linewidth=0.3,
               color=r['color'], s=45, zorder=3)
    lims = [min(y_t.min(), y_p.min()) - 0.3, max(y_t.max(), y_p.max()) + 0.3]
    ax.plot(lims, lims, 'r--', linewidth=1.4, label='Perfect fit')
    m, b = np.polyfit(y_t, y_p, 1)
    ax.plot(np.linspace(*lims, 100), m * np.linspace(*lims, 100) + b,
            color='gray', linewidth=1.1, linestyle=':', label='OLS trend')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f"{r['label']}\nRMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs Actual Impact Score — All Models', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Coefficient plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, r in zip(axes.flat, all_results):
    df_c = r['mc_coef_df'].copy()
    df_c['Display'] = df_c['Feature'].str.replace('^RAW_', '', regex=True)
    bar_clrs = ['#E74C3C' if c > 0 else '#3498DB' for c in df_c['Mean Coef']]
    xerr_lo  = (df_c['Mean Coef'] - df_c['95% CI Lower']).clip(lower=0)
    xerr_hi  = (df_c['95% CI Upper'] - df_c['Mean Coef']).clip(lower=0)
    tick_lbls = [f"{d}  ({p:.0f}%)" for d, p in zip(df_c['Display'], df_c['% Non-zero'])]
    ax.barh(range(len(df_c)), df_c['Mean Coef'], xerr=[xerr_lo, xerr_hi],
            color=bar_clrs, ecolor='black', capsize=4,
            edgecolor='black', linewidth=0.4, height=0.65)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(df_c))); ax.set_yticklabels(tick_lbls, fontsize=8)
    ax.set_xlabel('Mean Coefficient (± 95% CI)')
    ax.set_title(f"{r['label']}\nα={r['best_alpha']:.4f} | L1={r['best_l1_ratio']:.2f}")
    ax.grid(True, axis='x', alpha=0.3)
legend_patches = [Patch(facecolor='#E74C3C', edgecolor='black', label='Positive'),
                  Patch(facecolor='#3498DB', edgecolor='black', label='Negative')]
fig.legend(handles=legend_patches, loc='upper right', fontsize=9)
plt.suptitle(f'ElasticNet Coefficients — All Models (n={N_ITER} MC)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Permutation importance plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, r in zip(axes.flat, all_results):
    pd_imp = r['perm_df'].copy()
    pd_imp['Display'] = pd_imp['Feature'].str.replace('^RAW_', '', regex=True)
    top = pd_imp.head(16)
    ax.barh(top['Display'][::-1], top['Importance Mean'][::-1],
            xerr=top['Importance Std'][::-1].clip(lower=0),
            color=r['color'], ecolor='black', capsize=4, height=0.65,
            edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean Decrease in R²'); ax.set_title(f"{r['label']}")
    ax.tick_params(axis='y', labelsize=8); ax.grid(True, axis='x', alpha=0.3)
plt.suptitle('Permutation Feature Importance — All Models', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

for r in all_results:
    rows = model_data.loc[r['X_test'].index].copy()
    rows['Predicted_Impact'] = r['y_pred']
    rows['Actual_Impact']    = r['y_test']
    print(f"\nTop 10 by Predicted Impact — {r['label']}:")
    print(rows.sort_values('Predicted_Impact', ascending=False)
          .head(10)[['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR',
                     'Predicted_Impact', 'Actual_Impact']].to_string(index=False))

# ════════��══════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON — Paired Wilcoxon on MC R²
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("STATISTICAL COMPARISON — Paired Wilcoxon Tests on MC R²")
print("═" * 70)

n_comp = len(list(combinations(all_results, 2)))
stat_rows = []
for r1, r2 in combinations(all_results, 2):
    diff      = r1['mc_r2'] - r2['mc_r2']
    mean_diff = diff.mean()
    sd_diff   = diff.std()
    cohens_d  = mean_diff / sd_diff if sd_diff > 0 else 0.0
    stat_w, pval = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    pval_corr    = min(pval * n_comp, 1.0)
    stat_rows.append({
        'Comparison':     f"{r1['label']}  vs  {r2['label']}",
        'Mean ΔR²':       round(mean_diff, 4),
        '95% CI ΔR²':     f"[{np.percentile(diff,2.5):.4f}, {np.percentile(diff,97.5):.4f}]",
        "Cohen's d":      round(cohens_d, 3),
        'W':              int(stat_w),
        'p (Bonf.)':      round(pval_corr, 4),
        'Sig':            '***' if pval_corr < 0.001 else '**' if pval_corr < 0.01
                          else '*' if pval_corr < 0.05 else 'ns'
    })
stat_df = pd.DataFrame(stat_rows)
print(stat_df.to_string(index=False))

labels_nfl = [r['label'] for r in all_results]
n_m        = len(all_results)
delta_mat  = np.full((n_m, n_m), np.nan)
pval_mat   = np.full((n_m, n_m), np.nan)
for row in stat_rows:
    parts  = row['Comparison'].split('  vs  ')
    l1, l2 = parts[0].strip(), parts[1].strip()
    i = next(idx for idx, r in enumerate(all_results) if r['label'] == l1)
    j = next(idx for idx, r in enumerate(all_results) if r['label'] == l2)
    delta_mat[i, j] = row['Mean ΔR²'];  delta_mat[j, i] = -row['Mean ΔR²']
    pval_mat[i, j]  = row['p (Bonf.)']; pval_mat[j, i]  = row['p (Bonf.)']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(delta_mat, annot=True, fmt='.4f', cmap='coolwarm', center=0,
            xticklabels=labels_nfl, yticklabels=labels_nfl, linewidths=0.5, ax=axes[0])
axes[0].set_title('Mean ΔR² (row − column)')
axes[0].tick_params(axis='x', rotation=20)

annot_p = pd.DataFrame(pval_mat, index=labels_nfl, columns=labels_nfl).map(
    lambda v: f"{v:.4f}" if not np.isnan(v) else '—')
sns.heatmap(pval_mat, annot=annot_p, fmt='', cmap='YlOrRd_r', vmin=0, vmax=0.05,
            xticklabels=labels_nfl, yticklabels=labels_nfl, linewidths=0.5, ax=axes[1])
axes[1].set_title('Bonferroni p-value')
axes[1].tick_params(axis='x', rotation=20)
plt.suptitle('Pairwise Model Comparison — Paired Wilcoxon', fontsize=12)
plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE COMPARISON — NFL Outcomes
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_COLORS = ["#FF4747", "#199ACD", "#44AB59"]

model_data = model_data.merge(data[join_keys + ['Archetype']], on=join_keys, how='left')
arch_order  = sorted(model_data['Archetype'].dropna().unique())
arch_labels = [f'Archetype {int(a)}' for a in arch_order]

print(f"\nArchetype distribution in modelling dataset:")
print(model_data['Archetype'].value_counts().sort_index().to_string())

for r in all_results:
    model_data[f'PRED_{r["label"]}'] = r['en_final'].predict(
        model_data[r['feature_names']].values)

rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, r in zip(axes.flat, all_results):
    pred_col = f'PRED_{r["label"]}'
    groups   = [model_data[model_data['Archetype'] == a][pred_col].values for a in arch_order]
    parts    = ax.violinplot(groups, positions=range(len(arch_order)),
                             showmedians=True, widths=0.65)
    for pc, c in zip(parts['bodies'], ARCH_COLORS):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[part].set_color('black'); parts[part].set_linewidth(1.0)
    for idx, (grp, c) in enumerate(zip(groups, ARCH_COLORS)):
        jitter = rng.uniform(-0.12, 0.12, size=len(grp))
        ax.scatter(idx + jitter, grp, alpha=0.35, s=16, color=c, edgecolors='none', zorder=3)
    h_stat, p_kw = kruskal(*groups)
    sig_str = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'ns'
    ax.set_xticks(range(len(arch_order))); ax.set_xticklabels(arch_labels, fontsize=9)
    ax.set_ylabel('Predicted Impact Score')
    ax.set_title(f"{r['label']}\nKW: H={h_stat:.2f}, p={p_kw:.4f} {sig_str}")
    ax.grid(True, axis='y', alpha=0.3)
plt.suptitle('Predicted Impact Score by Archetype — All Models', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

act_groups = [model_data[model_data['Archetype'] == a]['IMPACT_SCORE'].dropna().values
              for a in arch_order]
h_act, p_act = kruskal(*act_groups)
sig_act = '***' if p_act < 0.001 else '**' if p_act < 0.01 else '*' if p_act < 0.05 else 'ns'

fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot(act_groups, positions=range(len(arch_order)),
                      showmedians=True, widths=0.65)
for pc, c in zip(parts['bodies'], ARCH_COLORS):
    pc.set_facecolor(c); pc.set_alpha(0.55)
for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
    parts[part].set_color('black'); parts[part].set_linewidth(1.0)
for idx, (grp, c) in enumerate(zip(act_groups, ARCH_COLORS)):
    jitter = rng.uniform(-0.12, 0.12, size=len(grp))
    ax.scatter(idx + jitter, grp, alpha=0.4, s=20, color=c, edgecolors='none', zorder=3)
ax.set_xticks(range(len(arch_order))); ax.set_xticklabels(arch_labels)
ax.set_ylabel('Actual Impact Score')
ax.set_title(f'Actual Impact Score by Archetype\nKW: H={h_act:.2f}, p={p_act:.4f} {sig_act}')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

arch_pairs   = list(combinations(arch_order, 2))
n_arch_pairs = len(arch_pairs)

print(f"\nActual Impact — Pairwise Mann-Whitney U (Bonferroni n={n_arch_pairs}):")
for a1, a2 in arch_pairs:
    g1 = model_data[model_data['Archetype'] == a1]['IMPACT_SCORE'].dropna()
    g2 = model_data[model_data['Archetype'] == a2]['IMPACT_SCORE'].dropna()
    u_s, p_r = mannwhitneyu(g1, g2, alternative='two-sided')
    p_b = min(p_r * n_arch_pairs, 1.0)
    sig = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else 'ns'
    print(f"  Arch {int(a1)} vs {int(a2)}: U={u_s:.0f}  p(Bonf)={p_b:.4f}  {sig}")

for r in all_results:
    pred_col = f'PRED_{r["label"]}'
    print(f"\n{r['label']} — Predicted Impact Pairwise MW-U (Bonf n={n_arch_pairs}):")
    for a1, a2 in arch_pairs:
        g1 = model_data[model_data['Archetype'] == a1][pred_col].dropna()
        g2 = model_data[model_data['Archetype'] == a2][pred_col].dropna()
        u_s, p_r = mannwhitneyu(g1, g2, alternative='two-sided')
        p_b = min(p_r * n_arch_pairs, 1.0)
        sig = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else 'ns'
        print(f"  Arch {int(a1)} vs {int(a2)}: U={u_s:.0f}  p(Bonf)={p_b:.4f}  {sig}")

print("\nActual Impact Score — Summary by Archetype:")
print(model_data.groupby('Archetype')['IMPACT_SCORE']
      .agg(N='count', Mean='mean', SD='std', Median='median',
           Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75))
      .round(3).to_string())

for r in all_results:
    pred_col = f'PRED_{r["label"]}'
    print(f"\nPredicted Impact — {r['label']}:")
    print(model_data.groupby('Archetype')[pred_col]
          .agg(N='count', Mean='mean', SD='std', Median='median')
          .round(3).to_string())

print("\n✅ Combined analysis complete.")