# Generated from: breast_cancer_remaining_work.ipynb
# Converted at: 2026-10-28T13:27:29.020Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Breast Cancer Classification — Remaining Work
# Self-contained: it reloads the data, re-creates the same split (seed 42) and re-trains the two baseline
# models, so you do **not** need the first notebook open. Run all cells top to bottom (about 1–2 minutes).
# 
# | Section | Remaining-work item |
# |---|---|
# | 1 | Is the accuracy gap real? McNemar's test (+ two stronger checks) |
# | 2 | Tune regularisation (C) and the decision threshold to favour malignant recall |
# | 3 | Third model: Random Forest, plus a SHAP view of feature effects |
# | 4 | Calibration plot and the final comparison of every model |
# | 5 | Save results and `requirements.txt` for GitHub |


# ---------------------------------------------------------------------------
# SECTION 0 — Setup (Google Drive on Colab; harmless elsewhere)
# ---------------------------------------------------------------------------
import os, sys, json, subprocess, warnings
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = '/content/drive/MyDrive/breastcancer-project'
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.chdir(PROJECT_DIR)
except ImportError:
    pass                                   # not on Colab: just use the current folder
os.makedirs("outputs", exist_ok=True)
print("Working directory:", os.getcwd())

try:
    import shap
except ImportError:                        # Colab usually needs this once
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])
    import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, fbeta_score,
                             log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
SEED = 42

# ---------------------------------------------------------------------------
# Data, split and the two baseline models (identical to the main notebook)
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")            # 0 = malignant, 1 = benign

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

lr = LogisticRegression(max_iter=5000, random_state=SEED).fit(Xtr_s, ytr)
gnb = GaussianNB().fit(Xtr_s, ytr)

pred_lr, pred_nb = lr.predict(Xte_s), gnb.predict(Xte_s)
print(f"Train {len(ytr)} / Test {len(yte)}")
print(f"Baseline accuracy  LR = {accuracy_score(yte, pred_lr):.4f}   GNB = {accuracy_score(yte, pred_nb):.4f}")
assert abs(accuracy_score(yte, pred_lr) - 0.9825) < 1e-3 and abs(accuracy_score(yte, pred_nb) - 0.9298) < 1e-3, \
    "Baselines differ from the main notebook: check the library versions / split"

# ## 1. Is the accuracy gap real?
# **1a. McNemar's test** looks only at the test patients where the two models *disagree*:
# * *b* = LR right, GNB wrong; *c* = LR wrong, GNB right.
# * Null hypothesis: the two models make errors at the same rate, so b and c are equally likely (each 50%).
# * With so few disagreements the **exact binomial** version is the right one (the chi-square version is shown for comparison).
# 
# **1b. Paired bootstrap** gives a 95% confidence interval for the accuracy difference.
# **1c. Repeated cross-validation** uses all 569 patients (not only 114) with the Nadeau–Bengio corrected t-test,
# which accounts for the folds sharing training data.


# 1a. McNemar's test -------------------------------------------------------
def mcnemar(name_a, ok_a, name_b, ok_b):
    ok_a, ok_b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    b = int((ok_a & ~ok_b).sum())          # A right, B wrong
    c = int((~ok_a & ok_b).sum())          # A wrong, B right
    n = b + c
    p_exact = stats.binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return {"pair": f"{name_a} vs {name_b}", "both_right": int((ok_a & ok_b).sum()),
            "only_first_right": b, "only_second_right": c, "both_wrong": int((~ok_a & ~ok_b).sum()),
            "p_exact": p_exact, "p_chi2_cc": stats.chi2.sf(chi2, 1) if n else 1.0}

ok_lr, ok_nb = (pred_lr == yte).to_numpy(), (pred_nb == yte).to_numpy()
mc = mcnemar("LR", ok_lr, "GNB", ok_nb)
print("2x2 table of correct/incorrect (rows: LR, columns: GNB)")
print(pd.DataFrame([[mc["both_right"], mc["only_first_right"]], [mc["only_second_right"], mc["both_wrong"]]],
                   index=["LR right", "LR wrong"], columns=["GNB right", "GNB wrong"]))
print(f"\nExact McNemar p-value           : {mc['p_exact']:.4f}")
print(f"Chi-square (continuity) p-value : {mc['p_chi2_cc']:.4f}")

# 1b. Paired bootstrap CI for the accuracy difference -----------------------
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(yte), size=(10_000, len(yte)))
boot = ok_lr[idx].mean(axis=1) - ok_nb[idx].mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Accuracy difference (LR - GNB) on the test set: {ok_lr.mean() - ok_nb.mean():+.4f}")
print(f"95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]   (an interval excluding 0 supports a real difference)")

# 1c. Repeated 5-fold CV + Nadeau-Bengio corrected resampled t-test ----------
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
mk_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
mk_nb = lambda: make_pipeline(StandardScaler(), GaussianNB())
acc_lr, acc_nb = [], []
for tr, te in rcv.split(X, y):
    acc_lr.append(accuracy_score(y.iloc[te], mk_lr().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])))
    acc_nb.append(accuracy_score(y.iloc[te], mk_nb().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])))
d = np.array(acc_lr) - np.array(acc_nb)
k = len(d)
t_stat = d.mean() / np.sqrt((1 / k + 1 / 4) * d.var(ddof=1))     # test/train size ratio = 1/4 for 5-fold
p_val = 2 * stats.t.sf(abs(t_stat), k - 1)
print(f"{k} resamples: mean accuracy  LR = {np.mean(acc_lr):.4f}   GNB = {np.mean(acc_nb):.4f}")
print(f"Mean paired difference = {d.mean():+.4f},  corrected t = {t_stat:.2f},  p = {p_val:.4g}")

# ### How to read Section 1 (for your report)
# * Report the **exact McNemar p-value** and the table of b and c (the disagreements).
# * With ~114 test patients the test has **low power**: a large p-value would mean "not enough evidence", not "no difference".
# * 1b and 1c are cross-checks. Agreement between all three is a stronger statement than any single p-value.
# * If you now have a p-value, update the Limitations text that says the gap was "not statistically tested".


# ## 2. Tune regularisation (C) and the decision threshold
# **Regularisation.** scikit-learn's default is C = 1.0 (L2 penalty; smaller C = stronger shrinkage of the weights).
# We search C on a log grid with 5-fold stratified CV **inside the training set only**, with scaling refitted in each fold.
# The primary score is log-loss (it rewards good *probabilities*, not only correct labels); AUC is recorded too.
# 
# **Threshold.** The default rule "predict malignant if P(malignant) ≥ 0.5" is not sacred. In screening a missed cancer is
# costlier than a false alarm, so we pick the threshold that maximises **F2** (recall weighted twice as heavily as precision)
# using *out-of-fold training predictions*, and only then apply it once to the test set. Malignant is the positive class here.


# 2a. Grid search over C -----------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])
grid = {"lr__C": np.logspace(-3, 3, 13)}
gs = GridSearchCV(pipe, grid, cv=cv, scoring={"logloss": "neg_log_loss", "auc": "roc_auc"},
                  refit="logloss", n_jobs=-1)
gs.fit(Xtr, ytr)
res = pd.DataFrame({"C": grid["lr__C"],
                    "cv_logloss": -gs.cv_results_["mean_test_logloss"],
                    "cv_auc": gs.cv_results_["mean_test_auc"]})
print(res.round(4).to_string(index=False))
best_C = gs.best_params_["lr__C"]
print(f"\nBest C by CV log-loss: {best_C:.4g}   (default was 1.0)")
TUNED_DIFFERS = not np.isclose(best_C, 1.0)
if not TUNED_DIFFERS:
    print("=> The grid confirms scikit-learn's default C = 1.0 is already the best value. That is a legitimate finding: report it.")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].semilogx(res.C, res.cv_logloss, "o-"); axes[0].axvline(best_C, color="red", ls="--", label=f"best C = {best_C:.3g}")
axes[0].axvline(1.0, color="grey", ls=":", label="default C = 1"); axes[0].set_xlabel("C"); axes[0].set_ylabel("CV log-loss (lower is better)"); axes[0].legend()
axes[1].semilogx(res.C, res.cv_auc, "o-", color="tab:green"); axes[1].axvline(best_C, color="red", ls="--")
axes[1].set_xlabel("C"); axes[1].set_ylabel("CV ROC-AUC")
plt.suptitle("Regularisation strength: 5-fold CV on the training set"); plt.tight_layout()
plt.savefig("outputs/tuning_C.png", dpi=150); plt.show()

# 2b. Threshold selection (out-of-fold, training set only) ---------------------
tuned = gs.best_estimator_                                   # already refit on the full training set
oof_mal = cross_val_predict(clone(tuned), Xtr, ytr, cv=cv, method="predict_proba")[:, 0]   # column 0 = P(malignant)
ytr_mal = (ytr == 0).astype(int)

rows = []
for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
    pr = (oof_mal >= t).astype(int)
    rows.append({"threshold": t, "recall_malignant": recall_score(ytr_mal, pr),
                 "precision_malignant": precision_score(ytr_mal, pr, zero_division=0), "F2": fbeta_score(ytr_mal, pr, beta=2)})
thr_df = pd.DataFrame(rows)
best_row = thr_df.sort_values(["F2"], ascending=False).assign(dist=lambda d_: (d_.threshold - 0.5).abs()) \
                 .sort_values(["F2", "dist"], ascending=[False, True]).iloc[0]
THR = float(best_row.threshold)
print(thr_df.round(3).to_string(index=False))
print(f"\nChosen threshold (max F2 on out-of-fold training predictions): {THR:.2f}")

plt.figure(figsize=(6.5, 4))
plt.plot(thr_df.threshold, thr_df.recall_malignant, "o-", label="recall (malignant)")
plt.plot(thr_df.threshold, thr_df.precision_malignant, "s-", label="precision (malignant)")
plt.plot(thr_df.threshold, thr_df.F2, "^--", label="F2")
plt.axvline(THR, color="red", ls="--", label=f"chosen = {THR:.2f}"); plt.axvline(0.5, color="grey", ls=":", label="default 0.5")
plt.xlabel("Threshold on P(malignant)"); plt.ylabel("Score"); plt.title("Threshold trade-off (out-of-fold, training set)")
plt.legend(); plt.tight_layout(); plt.savefig("outputs/threshold_tradeoff.png", dpi=150); plt.show()

# 2c. Apply to the TEST set once -------------------------------------------------
p_mal_te = tuned.predict_proba(Xte)[:, 0]
yte_mal = (yte == 0).astype(int)
out = []
for label, t in [("default 0.5", 0.5), (f"chosen {THR:.2f}", THR)]:
    pr = (p_mal_te >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte_mal, pr).ravel()
    out.append({"rule": label, "malignant_caught": int(tp), "malignant_MISSED": int(fn), "benign_flagged": int(fp),
                "recall_malignant": tp / (tp + fn), "precision_malignant": tp / (tp + fp) if tp + fp else np.nan})
cmp_thr = pd.DataFrame(out)
print(cmp_thr.round(4).to_string(index=False))
print(f"\nTuned model (C = {best_C:.3g}) test accuracy at 0.5 = {accuracy_score(yte, tuned.predict(Xte)):.4f}  "
      f"| default-C model = {accuracy_score(yte, pred_lr):.4f}")
print("Note: the test set has only 42 malignant patients, so differences of one or two patients are not conclusive.")

# ## 3. Third model — Random Forest — and a SHAP view of feature effects
# A Random Forest averages many decision trees. It needs no scaling and makes **no linear and no independence assumption**,
# so it is a useful contrast to both Logistic Regression and Naive Bayes.
# 
# **SHAP** assigns each feature a contribution to each individual prediction (Shapley values from game theory).
# Here a **positive SHAP value pushes the prediction toward malignant**. The bar chart is the mean absolute SHAP value per feature.


rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
pred_rf = rf.predict(Xte)
cv_auc_rf = cross_val_score(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
print(f"Random Forest  test accuracy = {accuracy_score(yte, pred_rf):.4f}   test AUC = {roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]):.4f}")
print(f"5-fold CV AUC = {cv_auc_rf.mean():.4f} (+/- {cv_auc_rf.std():.4f})")

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Xte)
# shap returns either a list [class0, class1] or an array (n, features, classes) depending on the version
sv_mal = sv[0] if isinstance(sv, list) else (sv[:, :, 0] if np.ndim(sv) == 3 else sv)   # class 0 = malignant
assert sv_mal.shape == Xte.shape, sv_mal.shape

plt.close("all")
shap.summary_plot(sv_mal, Xte, plot_type="bar", max_display=10, show=False)
plt.title("Mean |SHAP| — Random Forest (effect on P(malignant))"); plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight"); plt.show()

plt.close("all")
shap.summary_plot(sv_mal, Xte, max_display=10, show=False)
plt.title("SHAP values — red = high feature value; right = pushes toward malignant"); plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.show()

# Do the Random Forest and Logistic Regression agree about which features matter?
imp_rf = pd.Series(np.abs(sv_mal).mean(axis=0), index=X.columns).sort_values(ascending=False)
imp_lr = pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values(ascending=False)
top = pd.DataFrame({"RF (mean |SHAP|)": imp_rf.head(10).index, "LR (|coefficient|)": imp_lr.head(10).index})
print(top.to_string(index=False))
overlap = set(imp_rf.head(10).index) & set(imp_lr.head(10).index)
print(f"\nFeatures in both top-10 lists ({len(overlap)}): {sorted(overlap)}")
rho = stats.spearmanr(imp_rf.reindex(X.columns), imp_lr.reindex(X.columns)).statistic
print(f"Spearman rank correlation of the two importance rankings (all 30 features): {rho:.2f}")

# ### How to read Sections 2 and 3
# * If the best C equals 1.0, tuning simply **confirms the default**; say so rather than hiding it.
# * A lower threshold trades *more false alarms* for *fewer missed cancers*. Check on the test table whether it actually caught any extra malignant patient.
# * SHAP shows what the **Random Forest** relies on; the LR coefficients show what **Logistic Regression** relies on. Agreement between two very different models is evidence the features really matter.
# 
# ## 4. Calibration and the final comparison
# A **calibration (reliability) plot** asks whether predicted probabilities mean what they say: among patients given
# P(benign) ≈ 0.8, are about 80% really benign? Points on the diagonal = well calibrated. The **Brier score** (mean squared
# probability error) and **log-loss** summarise this; lower is better. With 114 test patients the curves are noisy.
# The last table runs McNemar for every pair. Three tests were run, so the **Bonferroni-adjusted** column is the cautious one;
# your planned comparison (LR vs GNB) is the unadjusted p-value from Section 1.


p_tuned = tuned.predict_proba(Xte)[:, 1]
probs = {"LR (default C)": lr.predict_proba(Xte_s)[:, 1], "Gaussian NB": gnb.predict_proba(Xte_s)[:, 1],
         "Random Forest": rf.predict_proba(Xte)[:, 1]}
preds = {"LR (default C)": pred_lr, "Gaussian NB": pred_nb, "Random Forest": pred_rf}
if TUNED_DIFFERS:                                                             # only add a separate row if tuning changed C
    probs[f"LR (tuned C={best_C:.3g})"] = p_tuned; preds[f"LR (tuned C={best_C:.3g})"] = tuned.predict(Xte)
CAL_MODELS = list(probs)                                                      # models shown on the calibration plot
probs[f"LR (tuned, thr {THR:.2f})"] = p_tuned                                 # same probabilities, different decision rule
preds[f"LR (tuned, thr {THR:.2f})"] = np.where(p_mal_te >= THR, 0, 1)         # decision from the chosen malignant threshold

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
for name in CAL_MODELS:
    frac, mean_p = calibration_curve(yte, probs[name], n_bins=5, strategy="quantile")
    ax.plot(mean_p, frac, "o-", label=name)
ax.set_xlabel("Mean predicted P(benign)"); ax.set_ylabel("Observed fraction benign"); ax.set_title("Calibration (reliability) plot, test set")
ax.legend(); plt.tight_layout(); plt.savefig("outputs/calibration.png", dpi=150); plt.show()

rows = []
for name, pr in preds.items():
    p = probs[name]
    tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()            # benign = positive class (as in the report)
    rows.append({"Model": name, "Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
                 "Recall": recall_score(yte, pr), "Specificity": tn / (tn + fp), "F1": f1_score(yte, pr),
                 "ROC-AUC": roc_auc_score(yte, p), "Brier": brier_score_loss(yte, p), "LogLoss": log_loss(yte, p),
                 "Malignant missed": int(fp)})
final = pd.DataFrame(rows).set_index("Model")
print(final.round(4).to_string())

# ROC overlay + pairwise McNemar for every model pair -----------------------------
plt.figure(figsize=(6, 5))
for name in ["LR (default C)", "Gaussian NB", "Random Forest"]:
    f, t_, _ = roc_curve(yte, probs[name]); plt.plot(f, t_, label=f"{name} (AUC = {roc_auc_score(yte, probs[name]):.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves, test set"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("outputs/roc_three_models.png", dpi=150); plt.show()

okay = {"LR": (pred_lr == yte).to_numpy(), "GNB": (pred_nb == yte).to_numpy(), "RF": (pred_rf == yte).to_numpy()}
pairs = [("LR", "GNB"), ("LR", "RF"), ("GNB", "RF")]
mc_all = pd.DataFrame([mcnemar(a, okay[a], b, okay[b]) for a, b in pairs])
mc_all["p_exact_bonferroni"] = (mc_all["p_exact"] * len(pairs)).clip(upper=1.0)     # 3 pairwise tests -> multiply by 3
print("only_first_right = first model right, second wrong;  only_second_right = the reverse")
print(mc_all.round(4).to_string(index=False))

# ## 5. Save results for the report and GitHub


final.round(4).to_csv("outputs/final_comparison.csv")
mc_all.round(4).to_csv("outputs/mcnemar_pairs.csv", index=False)
cmp_thr.round(4).to_csv("outputs/threshold_test_results.csv", index=False)
json.dump({"best_C": float(best_C), "threshold_max_F2": THR, "mcnemar_LR_vs_GNB_exact_p": float(mc["p_exact"]),
           "bootstrap_ci_acc_diff": [float(lo), float(hi)], "corrected_t_p": float(p_val)},
          open("outputs/extension_summary.json", "w"), indent=2)

from importlib.metadata import version
pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "shap"]
with open("requirements.txt", "w") as f:
    f.write("\n".join(f"{p}=={version(p)}" for p in pkgs) + "\n")
print(open("requirements.txt").read())
print("Saved files:", sorted(os.listdir("outputs")))