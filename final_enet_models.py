import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from factor_analyzer import FactorAnalyzer
import shap

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — Factor Analysis on PRE_NFL_DATA
# Identical pipeline to de_archetype_analysis.py so that the
# factors and loadings are directly comparable across analyses.
# ═══════════════════════════════════════════════════════════════

pre = pd.read_csv('/Users/bendoyle/Library/CloudStorage/OneDrive-TexasA&MUniversity/BD-JDT-TAMU-Sport-Data-Challenge-2026/'
                  'Post-Query Datasets/PRE_NFL_DATA.csv')

print(f"Total players before filtering: {pre.shape[0]}")
pre = pre[pre['COMBINE_TESTS_MISSING'] <= 3].copy()
print(f"Players after filtering (≤3 missing combine tests): {pre.shape[0]}")

# Derived velocity variables (identical to de_archetype_analysis.py)
pre['X40Y_VEL'] = 36.57 / pre['X40Y_SEC']
pre['X3CD_VEL'] = 27.43 / pre['X3CD_SEC']
pre['PROA_VEL'] = 18.28 / pre['PROA_SEC']
pre['mCODD']    = pre['PROA_SEC'] / pre['X40Y_SEC']

# Variable lists — identical to de_archetype_analysis.py
combine_variables        = ['VJ_IN', 'SLJ_IN', 'BENCH_REPS', 'P40', 'MANN_SLJP',
                            'X40Y_VEL', 'X3CD_VEL', 'PROA_VEL']
college_variables        = ['COLLEGE_SACKS', 'COLLEGE_FORCED_FUMBLES',
                            'COLLEGE_TACKLES_SOLO', 'COLLEGE_TACKLES_ASSISTED',
                            'COLLEGE_PASSES_DEFENDED', 'COLLEGE_INTERCEPTIONS']
anthropometric_variables = ['HEIGHT_IN', 'WEIGHT_LB']

# RF imputation on combine variables (identical to de_archetype_analysis.py)
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
    pre = random_forest_imputation(
        pre, var,
        [c for c in combine_variables if c != var] + anthropometric_variables
    )

# Standardize FA variables only — identical to de_archetype_analysis.py
fa_vars     = combine_variables + college_variables + anthropometric_variables
scaler      = StandardScaler()
data_scaled = scaler.fit_transform(pre[fa_vars])

# Store standardized raw features under new column names for Model B
# (RAW_ prefix avoids overwriting the un-scaled originals in pre)
raw_col_names     = [f'RAW_{v}' for v in fa_vars]
pre[raw_col_names] = data_scaled
raw_variables     = raw_col_names

# Scree plot
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

# 4-factor oblimin — identical to de_archetype_analysis.py
n_factors = 4
fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
fa.fit(data_scaled)

loadings_df = pd.DataFrame(
    fa.loadings_,
    index=fa_vars,
    columns=[f'Factor_{i+1}' for i in range(n_factors)]
)
print("\nFactor Loadings:")
print(loadings_df.round(3))

total_variance = np.sum(eigenvalues)
explained_variance_df = pd.DataFrame({
    'Explained Variance (%)': (eigenvalues[:n_factors] / total_variance * 100).round(2),
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

# Factor scores for every prospect in PRE_NFL_DATA
factor_scores   = fa.transform(data_scaled)
factor_score_df = pd.DataFrame(
    factor_scores, index=pre.index,
    columns=[f'Factor_{i+1}' for i in range(n_factors)]
)
pre             = pd.concat([pre, factor_score_df], axis=1)
factor_variables = list(factor_score_df.columns)

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — NFL Outcome Data
# Load nfl_data, filter to meaningful careers, build outcome.
# ═══════════════════════════════════════════════════════════════

nfl = pd.read_csv('/Users/bendoyle/Library/CloudStorage/OneDrive-TexasA&MUniversity/BD-JDT-TAMU-Sport-Data-Challenge-2026/'
                  'Post-Query Datasets/nfl_data.csv')

nfl = nfl[nfl['NFL_CAREER_GAMES_PLAYED'] >= 10].copy()
print(f"\nNFL players with ≥10 games: {nfl.shape[0]}")

# Outcome: log1p(sacks + TFL), z-scored
impact_raw   = nfl['NFL_CAREER_SACKS'].fillna(0) + nfl['NFL_CAREER_TFL'].fillna(0)
impact_log   = impact_raw.apply(lambda x: np.log1p(x) if x > 0 else 0)  # log1p for positive values, 0 for zero
impact_score = (impact_log - impact_log.mean()) / impact_log.std()
nfl['IMPACT_SCORE'] = impact_score

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Join factor scores → NFL outcomes
# Merge on name + draft year so we use the PRE-draft factors
# (fit on all prospects) to predict NFL career outcomes.
# ═══════════════════════════════════════════════════════════════

join_keys  = ['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR']
model_data = (nfl[join_keys + ['IMPACT_SCORE']]
              .merge(pre[join_keys + factor_variables + raw_variables], on=join_keys, how='inner'))

print(f"Modeling dataset: {model_data.shape[0]} players "
      f"(pre-draft factors matched to NFL outcomes)")

plt.figure(figsize=(8, 5))
sns.histplot(model_data['IMPACT_SCORE'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Impact Score (log1p, z-scored)')
plt.xlabel('Impact Score')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# shared settings for both models
y          = model_data['IMPACT_SCORE']
l1_ratios  = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
alpha_grid = np.logspace(-4, 0, 50)
N_ITER     = 1000


def run_enet_model(X, feature_names, label, color):
    """Full ElasticNet pipeline: CV → Monte Carlo → final model.
    Returns a dict of results needed for comparison plots."""
    # ── ElasticNetCV ──
    en_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alpha_grid,
                         cv=5, max_iter=10000, random_state=42, n_jobs=-1)
    en_cv.fit(X, y)
    best_alpha    = en_cv.alpha_
    best_l1_ratio = en_cv.l1_ratio_
    print(f"\n{label} — ElasticNetCV: α={best_alpha:.6f}  L1={best_l1_ratio:.2f}")

    # ── Monte Carlo ──
    mc_coefs = np.zeros((N_ITER, len(feature_names)))
    mc_rmse  = np.zeros(N_ITER)
    mc_r2    = np.zeros(N_ITER)
    print(f"Running {N_ITER} Monte Carlo iterations ({label})...")
    for i in range(N_ITER):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=i)
        en = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                        max_iter=10000, random_state=i)
        en.fit(X_tr, y_tr)
        y_pred       = en.predict(X_te)
        mc_coefs[i]  = en.coef_
        mc_rmse[i]   = np.sqrt(mean_squared_error(y_te, y_pred))
        mc_r2[i]     = r2_score(y_te, y_pred)
    print("Done.")

    # ── Summary table ──
    mc_coef_df = pd.DataFrame({
        'Feature':      feature_names,
        'Mean Coef':    mc_coefs.mean(axis=0).round(4),
        'SD':           mc_coefs.std(axis=0).round(4),
        '95% CI Lower': np.percentile(mc_coefs, 2.5,  axis=0).round(4),
        '95% CI Upper': np.percentile(mc_coefs, 97.5, axis=0).round(4),
        '% Non-zero':   ((mc_coefs != 0).mean(axis=0) * 100).round(1)
    }).sort_values('Mean Coef', key=abs, ascending=False).reset_index(drop=True)

    print(f"\n── {label}: Monte Carlo Coefficient Summary ──")
    print(mc_coef_df.to_string(index=False))
    print(f"\n── {label}: Performance ({N_ITER} iterations) ──")
    for mname, vals in [('RMSE', mc_rmse), ('R²', mc_r2)]:
        print(f"  {mname}: Mean={vals.mean():.4f}  SD={vals.std():.4f}  "
              f"95% CI [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")

    # ── Final model seed=42 ──
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    en_final = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                          max_iter=10000, random_state=42)
    en_final.fit(X_tr, y_tr)
    y_pred_f   = en_final.predict(X_te)
    rmse_f     = np.sqrt(mean_squared_error(y_te, y_pred_f))
    r2_f       = r2_score(y_te, y_pred_f)
    print(f"\n{label} final (seed=42) — RMSE: {rmse_f:.4f} | R²: {r2_f:.4f}")

    # Permutation importance
    perm_res = permutation_importance(en_final, X_te, y_te,
                                      n_repeats=30, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        'Feature':         feature_names,
        'Importance Mean': perm_res.importances_mean,
        'Importance Std':  perm_res.importances_std
    }).sort_values('Importance Mean', ascending=False).reset_index(drop=True)
    print(f"\n{label} — Permutation Importances:")
    print(perm_df.to_string(index=False))

    return dict(
        label=label, color=color,
        mc_r2=mc_r2, mc_rmse=mc_rmse,
        en_final=en_final, best_alpha=best_alpha, best_l1_ratio=best_l1_ratio,
        X_test=X_te, y_test=y_te, y_pred=y_pred_f,
        rmse=rmse_f, r2=r2_f,
        perm_df=perm_df, mc_coefs=mc_coefs, mc_coef_df=mc_coef_df,
        feature_names=feature_names
    )


from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge

def run_rfecv(X_all, variables, color, title_tag):
    """Run RFECV on X_all, print results, plot CV R² curve, return selected variable list."""
    rfecv = RFECV(estimator=Ridge(alpha=1.0), step=1, cv=5, scoring='r2', n_jobs=-1)
    rfecv.fit(X_all, y)
    selected  = [v for v, s in zip(variables, rfecv.support_) if s]
    eliminated = [v for v, s in zip(variables, rfecv.support_) if not s]
    print(f"\nRFECV ({title_tag}): {rfecv.n_features_} / {len(variables)} features selected")
    print(f"  Selected  : {selected}")
    print(f"  Eliminated: {eliminated}")
    n_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
    mu  = rfecv.cv_results_['mean_test_score']
    sig = rfecv.cv_results_['std_test_score']
    plt.figure(figsize=(9, 4))
    plt.plot(n_range, mu, marker='o', color=color)
    plt.fill_between(n_range, mu - sig, mu + sig, alpha=0.25, color=color)
    plt.axvline(rfecv.n_features_, color='red', linestyle='--',
                label=f'Optimal = {rfecv.n_features_} features')
    plt.xlabel('Number of Features')
    plt.ylabel('CV R²')
    plt.title(f'RFECV — {title_tag}: CV R² vs Feature Count')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return selected


# ═══════════════════════════════════════════════════════════════
# MODEL A — EFA Factors, standard ElasticNet
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("MODEL A — EFA Factors (standard ElasticNet)")
print("═"*60)
res_a = run_enet_model(
    model_data[factor_variables], factor_variables,
    label='Model A (Factors)', color='steelblue'
)

# ═══════════════════════════════════════════════════════════════
# MODEL B — EFA Factors, ElasticNet + RFECV pre-selection
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("MODEL B — EFA Factors + RFECV")
print("═"*60)
selected_factors = run_rfecv(
    model_data[factor_variables], factor_variables,
    color='steelblue', title_tag='Model B (Factors + RFECV)'
)
res_b = run_enet_model(
    model_data[selected_factors], selected_factors,
    label='Model B (Factors + RFECV)', color='#1A6FA8'
)

# ═══════════════════════════════════════════════════════════════
# MODEL C — Raw Features, standard ElasticNet
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("MODEL C — Raw Features (standard ElasticNet)")
print("═"*60)
res_c = run_enet_model(
    model_data[raw_variables], raw_variables,
    label='Model C (Raw)', color='darkorange'
)

# ═══════════════════════════════════════════════════════════════
# MODEL D — Raw Features, ElasticNet + RFECV pre-selection
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("MODEL D — Raw Features + RFECV")
print("═"*60)
selected_raw = run_rfecv(
    model_data[raw_variables], raw_variables,
    color='darkorange', title_tag='Model D (Raw + RFECV)'
)
res_d = run_enet_model(
    model_data[selected_raw], selected_raw,
    label='Model D (Raw + RFECV)', color='#B85C00'
)

# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON — A / B / C / D
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("MODEL COMPARISON")
print("═"*60)

all_results = [res_a, res_b, res_c, res_d]

comp_df = pd.DataFrame({
    'Model':             [r['label'] for r in all_results],
    'N Features':        [len(r['feature_names']) for r in all_results],
    'Mean R²':           [round(r['mc_r2'].mean(), 4)  for r in all_results],
    'SD R²':             [round(r['mc_r2'].std(), 4)   for r in all_results],
    'R² CI Lower':       [round(np.percentile(r['mc_r2'], 2.5), 4)  for r in all_results],
    'R² CI Upper':       [round(np.percentile(r['mc_r2'], 97.5), 4) for r in all_results],
    'Mean RMSE':         [round(r['mc_rmse'].mean(), 4) for r in all_results],
    'SD RMSE':           [round(r['mc_rmse'].std(), 4)  for r in all_results],
    'Final R² (s=42)':   [round(r['r2'], 4)   for r in all_results],
    'Final RMSE (s=42)': [round(r['rmse'], 4) for r in all_results],
})
print(comp_df.to_string(index=False))

# 1. Overlapping R² distributions
plt.figure(figsize=(12, 5))
for r in all_results:
    sns.histplot(r['mc_r2'], bins=40, kde=True, alpha=0.35,
                 color=r['color'], label=f"{r['label']} (μ={r['mc_r2'].mean():.3f})")
    plt.axvline(r['mc_r2'].mean(), color=r['color'], linestyle='--', linewidth=1.5)
plt.xlabel('R²')
plt.title(f'R² Distribution — All Models ({N_ITER} Monte Carlo iterations)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# 2. Overlapping RMSE distributions
plt.figure(figsize=(12, 5))
for r in all_results:
    sns.histplot(r['mc_rmse'], bins=40, kde=True, alpha=0.35,
                 color=r['color'], label=f"{r['label']} (μ={r['mc_rmse'].mean():.3f})")
    plt.axvline(r['mc_rmse'].mean(), color=r['color'], linestyle='--', linewidth=1.5)
plt.xlabel('RMSE')
plt.title(f'RMSE Distribution — All Models ({N_ITER} Monte Carlo iterations)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# 3. Summary bar chart — Mean R² ± 95% CI
fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(all_results))
means = [r['mc_r2'].mean() for r in all_results]
lo    = [r['mc_r2'].mean() - np.percentile(r['mc_r2'], 2.5)  for r in all_results]
hi    = [np.percentile(r['mc_r2'], 97.5) - r['mc_r2'].mean() for r in all_results]
ax.bar(x, means, yerr=[lo, hi], color=[r['color'] for r in all_results],
       edgecolor='black', linewidth=0.6, capsize=6, width=0.55)
ax.set_xticks(x)
ax.set_xticklabels([r['label'] for r in all_results], rotation=15, ha='right')
ax.set_ylabel('Mean R² (95% CI)')
ax.set_title(f'Model Comparison — Mean R² ± 95% CI ({N_ITER} iterations)')
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

from matplotlib.patches import Patch

# ── Summary Figure 1: Predicted vs. Actual (2×2) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, r in zip(axes.flat, all_results):
    y_t, y_p = r['y_test'], r['y_pred']
    ax.scatter(y_t, y_p, alpha=0.6, edgecolors='black', linewidth=0.3,
               color=r['color'], s=45, zorder=3)
    lims = [min(y_t.min(), y_p.min()) - 0.3, max(y_t.max(), y_p.max()) + 0.3]
    ax.plot(lims, lims, 'r--', linewidth=1.4, label='Perfect fit', zorder=2)
    m, b = np.polyfit(y_t, y_p, 1)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, m * x_fit + b, color='gray', linewidth=1.1,
            linestyle=':', label='OLS trend', zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel('Actual Impact Score', fontsize=10)
    ax.set_ylabel('Predicted Impact Score', fontsize=10)
    ax.set_title(
        f"{r['label']}\n"
        f"seed=42: RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}   ",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs. Actual Impact Score — All Models (seed=42 test split)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Summary Figure 2: ElasticNet Coefficients (2×2) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, r in zip(axes.flat, all_results):
    df_c = r['mc_coef_df'].copy()
    df_c['Display'] = df_c['Feature'].str.replace('^RAW_', '', regex=True)
    bar_clrs = ['#E74C3C' if c > 0 else '#3498DB' for c in df_c['Mean Coef']]
    xerr_lo  = (df_c['Mean Coef'] - df_c['95% CI Lower']).clip(lower=0)
    xerr_hi  = (df_c['95% CI Upper'] - df_c['Mean Coef']).clip(lower=0)
    tick_lbls = [f"{d}  ({p:.0f}%)" for d, p in
                 zip(df_c['Display'], df_c['% Non-zero'])]
    ax.barh(range(len(df_c)), df_c['Mean Coef'],
            xerr=[xerr_lo, xerr_hi],
            color=bar_clrs, ecolor='black', capsize=4,
            edgecolor='black', linewidth=0.4, height=0.65)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(df_c)))
    ax.set_yticklabels(tick_lbls, fontsize=8)
    ax.set_xlabel('Mean Coefficient (± 95% CI)', fontsize=9)
    ax.set_title(
        f"{r['label']}\nα={r['best_alpha']:.4f}  |  L1={r['best_l1_ratio']:.2f}",
        fontsize=10
    )
    ax.grid(True, axis='x', alpha=0.3)
legend_patches = [
    Patch(facecolor='#E74C3C', edgecolor='black', label='Positive coefficient'),
    Patch(facecolor='#3498DB', edgecolor='black', label='Negative coefficient'),
]
fig.legend(handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.9)
plt.suptitle(
    f'ElasticNet Coefficients — All Models\n'
    f'(Mean ± 95% CI, n={N_ITER} MC iterations; % = iterations with non-zero coefficient)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.show()

# ── Summary Figure 3: Permutation Importance (2×2) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, r in zip(axes.flat, all_results):
    pd_imp = r['perm_df'].copy()
    pd_imp['Display'] = pd_imp['Feature'].str.replace('^RAW_', '', regex=True)
    top = pd_imp.head(16)
    ax.barh(top['Display'][::-1], top['Importance Mean'][::-1],
            xerr=top['Importance Std'][::-1].clip(lower=0),
            color=r['color'], ecolor='black', capsize=4,
            height=0.65, edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean Decrease in R² (Permutation)', fontsize=9)
    ax.set_title(f"{r['label']} — Permutation Feature Importance", fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, axis='x', alpha=0.3)
plt.suptitle('Permutation Feature Importance — All Models (seed=42 test set, 30 repeats)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Top 10 predicted players — all models
for r in all_results:
    rows = model_data.loc[r['X_test'].index].copy()
    rows['Predicted_Impact'] = r['y_pred']
    rows['Actual_Impact']    = r['y_test']
    print(f"\nTop 10 by Predicted Impact — {r['label']}:")
    print(rows.sort_values('Predicted_Impact', ascending=False)
          .head(10)[['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR', 'Predicted_Impact', 'Actual_Impact']]
          .to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON OF MODEL PERFORMANCE
#
# Because every Monte Carlo iteration uses the same random_state=i,
# all four models see the exact same train/test split at each
# iteration → R² values are PAIRED across models.
#
# Paired Wilcoxon signed-rank test on per-iteration R² differences:
#   - Non-parametric (no normality assumption on differences)
#   - More powerful than independent-samples tests for paired data
#   - 6 pairwise comparisons → Bonferroni correction (α / 6)
#
# Effect size: Cohen's d on the paired differences
#   d = mean_diff / sd_diff  (paired Cohen's d)
# ═══════════════════════════════════════════════════════════════
from scipy.stats import wilcoxon
from itertools import combinations

print("\n" + "═"*70)
print("STATISTICAL COMPARISON — Paired Wilcoxon Tests on MC R²")
print("(Bonferroni-corrected for 6 pairwise comparisons)")
print("═"*70)

n_comparisons = len(list(combinations(all_results, 2)))
alpha_corrected = 0.05 / n_comparisons

stat_rows = []
for r1, r2 in combinations(all_results, 2):
    diff       = r1['mc_r2'] - r2['mc_r2']
    mean_diff  = diff.mean()
    sd_diff    = diff.std()
    ci_lo      = np.percentile(diff, 2.5)
    ci_hi      = np.percentile(diff, 97.5)
    cohens_d   = mean_diff / sd_diff if sd_diff > 0 else 0.0
    stat, pval = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    pval_corr  = min(pval * n_comparisons, 1.0)   # Bonferroni cap at 1.0
    stat_rows.append({
        'Comparison':        f"{r1['label']}  vs  {r2['label']}",
        'Mean ΔR²':          round(mean_diff, 4),
        '95% CI ΔR²':        f"[{ci_lo:.4f}, {ci_hi:.4f}]",
        "Cohen's d":         round(cohens_d, 3),
        'W statistic':       int(stat),
        'p (raw)':           round(pval, 4),
        'p (Bonferroni)':    round(pval_corr, 4),
        'Significant':       '***' if pval_corr < 0.001 else
                             '**'  if pval_corr < 0.01  else
                             '*'   if pval_corr < 0.05  else 'ns'
    })

stat_df = pd.DataFrame(stat_rows)
pd.set_option('display.max_colwidth', 60)
print(stat_df.to_string(index=False))
print(f"\nα (Bonferroni-corrected) = {alpha_corrected:.4f}  |  "
      f"* p<.05  ** p<.01  *** p<.001  ns = not significant")

# Heatmap of mean ΔR² (upper triangle) and −log10(p_corrected) (lower triangle)
labels     = [r['label'] for r in all_results]
n          = len(all_results)
delta_mat  = np.full((n, n), np.nan)
pval_mat   = np.full((n, n), np.nan)

for row in stat_rows:
    parts    = row['Comparison'].split('  vs  ')
    l1, l2   = parts[0].strip(), parts[1].strip()
    i = next(idx for idx, r in enumerate(all_results) if r['label'] == l1)
    j = next(idx for idx, r in enumerate(all_results) if r['label'] == l2)
    delta_mat[i, j] = row['Mean ΔR²']
    pval_mat[i, j]  = row['p (Bonferroni)']
    delta_mat[j, i] = -row['Mean ΔR²']
    pval_mat[j, i]  = row['p (Bonferroni)']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean ΔR²
sns.heatmap(delta_mat, annot=True, fmt='.4f', cmap='coolwarm', center=0,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=axes[0])
axes[0].set_title('Mean ΔR² (row − column)\nPositive = row model is better')
axes[0].tick_params(axis='x', rotation=20)

# Right: Bonferroni-corrected p-value
annot_p = pd.DataFrame(pval_mat, index=labels, columns=labels).applymap(
    lambda v: f"{v:.4f}" if not np.isnan(v) else '—'
)
sns.heatmap(pval_mat, annot=annot_p, fmt='', cmap='YlOrRd_r', vmin=0, vmax=0.05,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=axes[1])
axes[1].set_title('Bonferroni-corrected p-value\n(dark = more significant)')
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle('Pairwise Model Comparison — Paired Wilcoxon (1000 MC iterations)', fontsize=12)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# PHASE 5 — ARCHETYPE COMPARISON
#
# Assign GMM archetypes (identical 3-cluster pipeline from
# de_archetype_analysis.py) using the factor scores already
# computed above, then compare both actual and predicted
# Impact Scores across archetype groups for all four models.
# ═══════════════════════════════════════════════════════════════

from sklearn.mixture import GaussianMixture
from scipy.stats import kruskal, mannwhitneyu

print("\n" + "═"*60)
print("PHASE 5 — ARCHETYPE COMPARISON")
print("═"*60)

ARCH_N       = 3
ARCH_COLORS  = ["#FF4747", "#199ACD", "#44AB59"]

# ── Fit GMM on the same factor scores used in de_archetype_analysis.py ──
gmm_arch = GaussianMixture(n_components=ARCH_N, random_state=42)
gmm_arch.fit(factor_score_df[factor_variables])
pre['Archetype'] = gmm_arch.predict(factor_score_df[factor_variables])

# ── Merge archetype labels into model_data ──
model_data = model_data.merge(
    pre[join_keys + ['Archetype']], on=join_keys, how='left'
)
arch_order = sorted(model_data['Archetype'].dropna().unique())
arch_labels = [f'Archetype {int(a)}' for a in arch_order]

print(f"\nArchetype distribution in modelling dataset:")
print(model_data['Archetype'].value_counts().sort_index().to_string())

# ── Predict on full model_data for every model ──
for r in all_results:
    X_full = model_data[r['feature_names']].values
    model_data[f'PRED_{r["label"]}'] = r['en_final'].predict(X_full)

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Predicted Impact Score by Archetype (2×2)
# One panel per model; violins + jittered points + K-W stat
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
rng = np.random.default_rng(42)

for ax, r in zip(axes.flat, all_results):
    pred_col = f'PRED_{r["label"]}'
    plot_df  = model_data[['Archetype', pred_col]].dropna()
    groups   = [plot_df[plot_df['Archetype'] == a][pred_col].values for a in arch_order]

    # violin
    parts = ax.violinplot(groups, positions=range(len(arch_order)),
                          showmedians=True, widths=0.65)
    for pc, c in zip(parts['bodies'], ARCH_COLORS):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[part].set_color('black'); parts[part].set_linewidth(1.0)

    # jitter
    for idx, (grp, c) in enumerate(zip(groups, ARCH_COLORS)):
        jitter = rng.uniform(-0.12, 0.12, size=len(grp))
        ax.scatter(idx + jitter, grp, alpha=0.35, s=16,
                   color=c, edgecolors='none', zorder=3)

    # Kruskal-Wallis
    h_stat, p_kw = kruskal(*groups)
    sig_str = ('***' if p_kw < 0.001 else '**' if p_kw < 0.01
               else '*' if p_kw < 0.05 else 'ns')

    ax.set_xticks(range(len(arch_order)))
    ax.set_xticklabels(arch_labels, fontsize=9)
    ax.set_ylabel('Predicted Impact Score', fontsize=9)
    ax.set_title(
        f"{r['label']}\nKruskal-Wallis: H={h_stat:.2f},  p={p_kw:.4f}  {sig_str}",
        fontsize=10
    )
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle(
    'Predicted Impact Score by Archetype — All Models\n'
    '(GMM 3-cluster solution on EFA factor scores)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Actual Impact Score by Archetype (single panel)
# ─────────────────────────────────────────────────────────────
act_groups = [model_data[model_data['Archetype'] == a]['IMPACT_SCORE'].dropna().values
              for a in arch_order]
h_act, p_act = kruskal(*act_groups)
sig_act = ('***' if p_act < 0.001 else '**' if p_act < 0.01
           else '*' if p_act < 0.05 else 'ns')

fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot(act_groups, positions=range(len(arch_order)),
                      showmedians=True, widths=0.65)
for pc, c in zip(parts['bodies'], ARCH_COLORS):
    pc.set_facecolor(c); pc.set_alpha(0.55)
for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
    parts[part].set_color('black'); parts[part].set_linewidth(1.0)
for idx, (grp, c) in enumerate(zip(act_groups, ARCH_COLORS)):
    jitter = rng.uniform(-0.12, 0.12, size=len(grp))
    ax.scatter(idx + jitter, grp, alpha=0.4, s=20,
               color=c, edgecolors='none', zorder=3)

ax.set_xticks(range(len(arch_order)))
ax.set_xticklabels(arch_labels, fontsize=10)
ax.set_ylabel('Actual Impact Score (log1p, z-scored)', fontsize=10)
ax.set_title(
    f'Actual Impact Score by Archetype\n'
    f'Kruskal-Wallis: H={h_act:.2f},  p={p_act:.4f}  {sig_act}',
    fontsize=12
)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────
# Pairwise Mann-Whitney U (Bonferroni) — Actual Impact Score
# ─────────────────────────────────────────────────────────────
arch_pairs   = list(combinations(arch_order, 2))
n_arch_pairs = len(arch_pairs)

print(f"\nActual Impact Score — Pairwise Mann-Whitney U "
      f"(Bonferroni-corrected for {n_arch_pairs} comparisons):")
print(f"{'Comparison':<38} {'U':>8} {'p (raw)':>10} {'p (Bonf.)':>12} {'Sig':>4}")
print("─" * 76)
for a1, a2 in arch_pairs:
    g1 = model_data[model_data['Archetype'] == a1]['IMPACT_SCORE'].dropna()
    g2 = model_data[model_data['Archetype'] == a2]['IMPACT_SCORE'].dropna()
    u_s, p_r = mannwhitneyu(g1, g2, alternative='two-sided')
    p_b = min(p_r * n_arch_pairs, 1.0)
    sig = ('***' if p_b < 0.001 else '**' if p_b < 0.01
           else '*' if p_b < 0.05 else 'ns')
    cmp = f'Archetype {int(a1)} vs Archetype {int(a2)}'
    print(f"{cmp:<38} {u_s:>8.0f} {p_r:>10.4f} {p_b:>12.4f} {sig:>4}")

# ─────────────────────────────────────────────────────────────
# Pairwise Mann-Whitney U (Bonferroni) — Predicted, each model
# ─────────────────────────────────────────────────────────────
for r in all_results:
    pred_col = f'PRED_{r["label"]}'
    print(f"\n{r['label']} — Predicted Impact Score Pairwise Mann-Whitney U "
          f"(Bonferroni n={n_arch_pairs}):")
    print(f"{'Comparison':<38} {'U':>8} {'p (raw)':>10} {'p (Bonf.)':>12} {'Sig':>4}")
    print("─" * 76)
    for a1, a2 in arch_pairs:
        g1 = model_data[model_data['Archetype'] == a1][pred_col].dropna()
        g2 = model_data[model_data['Archetype'] == a2][pred_col].dropna()
        u_s, p_r = mannwhitneyu(g1, g2, alternative='two-sided')
        p_b = min(p_r * n_arch_pairs, 1.0)
        sig = ('***' if p_b < 0.001 else '**' if p_b < 0.01
               else '*' if p_b < 0.05 else 'ns')
        cmp = f'Archetype {int(a1)} vs Archetype {int(a2)}'
        print(f"{cmp:<38} {u_s:>8.0f} {p_r:>10.4f} {p_b:>12.4f} {sig:>4}")

# ─────────────────────────────────────────────────────────────
# Summary stats — Actual + Predicted by Archetype
# ─────────────────────────────────────────────────────────────
print("\nActual Impact Score — Summary by Archetype:")
print(model_data.groupby('Archetype')['IMPACT_SCORE']
      .agg(N='count', Mean='mean', SD='std', Median='median',
           Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75))
      .round(3).to_string())

for r in all_results:
    pred_col = f'PRED_{r["label"]}'
    print(f"\nPredicted Impact Score — {r['label']}:")
    print(model_data.groupby('Archetype')[pred_col]
          .agg(N='count', Mean='mean', SD='std', Median='median')
          .round(3).to_string())
