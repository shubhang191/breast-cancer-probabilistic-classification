# Generated from: breast_cancer_classification (1).ipynb
# Converted at: 2026-10-28T13:25:18.262Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Breast Cancer Classification using Probabilistic Machine Learning
# **Track 6 — AI, Machine Learning & Probability**
# 
# Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), UCI ML Repository
# (bundled with scikit-learn as `load_breast_cancer`).
# 
# Probability concepts demonstrated: Conditional Probability, Gaussian
# Distribution, Maximum Likelihood Estimation, Log-Odds, Bayesian Inference.


# ---------------------------------------------------------------------------
# SECTION 0 — Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

import os
os.makedirs("outputs", exist_ok=True)   # where plots are saved for the report

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# SECTION 1 — Load the dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target          # 0 = malignant, 1 = benign

print("Shape:", df.shape)
print("\nTarget encoding:", dict(enumerate(data.target_names)))
print("\nClass counts:")
print(df["diagnosis"].value_counts().rename({0: "malignant", 1: "benign"}))
print("\nMissing values:", int(df.isnull().sum().sum()))

df.head()

df.describe().T[["mean", "std", "min", "max"]].head(10)

# ---------------------------------------------------------------------------
# SECTION 2 — Exploratory Data Analysis
# ---------------------------------------------------------------------------
# 2.1 Class balance — motivates why accuracy alone is not enough
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["diagnosis"].map({0: "Malignant", 1: "Benign"}).value_counts()
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_title("Class Distribution")
ax.set_ylabel("Number of samples")
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()

imbalance = counts.max() / counts.sum()
print(f"Majority class share: {imbalance:.1%}")
print("=> A model predicting 'benign' for everything would already score "
      f"{imbalance:.1%} accuracy. This is why precision/recall are reported.")

# 2.2 GAUSSIAN DISTRIBUTION — evidence that features are approximately normal.
#     This justifies (a) StandardScaler and (b) the Gaussian Naive Bayes model.
gaussian_features = ["mean radius", "mean texture", "mean smoothness", "mean symmetry"]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, feat in zip(axes.flatten(), gaussian_features):
    sns.histplot(df[feat], kde=True, ax=ax, bins=30)
    mu, sigma = df[feat].mean(), df[feat].std()
    ax.set_title(f"{feat}\nmean={mu:.3f}, std={sigma:.3f}")
plt.suptitle("Feature Distributions with KDE Overlay (Gaussian Check)")
plt.tight_layout()
plt.savefig("outputs/gaussian_distributions.png", dpi=150)
plt.show()

print("Distribution parameters (used in the report's Gaussian discussion):")
for feat in gaussian_features:
    print(f"  {feat:<20} mu = {df[feat].mean():8.4f}   sigma = {df[feat].std():8.4f}")

# 2.3 Separation between classes — boxplots show these features carry signal
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, feat in zip(axes, ["mean radius", "mean concave points", "worst perimeter"]):
    sns.boxplot(data=df, x="diagnosis", y=feat, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_xlabel("")
plt.suptitle("Feature Separation by Diagnosis")
plt.tight_layout()
plt.savefig("outputs/feature_boxplots.png", dpi=150)
plt.show()

# 2.4 Correlation heatmap — motivates the note on multicollinearity
subset = [c for c in df.columns if c.startswith("mean")] + ["diagnosis"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[subset].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation Matrix (mean-value features)")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# SECTION 3 — Train/test split and feature scaling
# ---------------------------------------------------------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# stratify=y keeps the malignant/benign ratio identical in both splits.
print(f"Training samples: {len(X_train)}   Test samples: {len(X_test)}")
print(f"Train class balance: {y_train.mean():.4f}   "
      f"Test class balance: {y_test.mean():.4f}")

# Scaler is FIT on training data only, then APPLIED to test data.
# Fitting on the full dataset would leak test information into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAfter scaling — training set mean ~0, std ~1:")
print(f"  mean = {X_train_scaled.mean():.6f}   std = {X_train_scaled.std():.6f}")

# ---------------------------------------------------------------------------
# SECTION 4 — Model 1: Logistic Regression
# ---------------------------------------------------------------------------
# Fitted by MAXIMUM LIKELIHOOD ESTIMATION: the solver searches for the
# coefficient vector that maximises the log-likelihood of the observed labels,
# which is equivalent to minimising log-loss (binary cross-entropy).
log_reg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]   # P(benign | features)

print("LOGISTIC REGRESSION")
print(classification_report(y_test, y_pred_lr,
                           target_names=["Malignant", "Benign"]))

# ---------------------------------------------------------------------------
# SECTION 5 — Model 2: Gaussian Naive Bayes
# ---------------------------------------------------------------------------
# Applies BAYES' THEOREM with a GAUSSIAN likelihood for each feature:
#   P(class | x) proportional to P(class) * product_i P(x_i | class)
# where each P(x_i | class) is modelled as a normal distribution.
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

y_pred_nb = gnb.predict(X_test_scaled)
y_proba_nb = gnb.predict_proba(X_test_scaled)[:, 1]

print("GAUSSIAN NAIVE BAYES")
print(classification_report(y_test, y_pred_nb,
                           target_names=["Malignant", "Benign"]))

# ---------------------------------------------------------------------------
# SECTION 6 — Evaluation
# ---------------------------------------------------------------------------
# 6.1 Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, preds) in zip(axes, [("Logistic Regression", y_pred_lr),
                                    ("Gaussian Naive Bayes", y_pred_nb)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
plt.show()

# 6.2 ROC curves
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, proba in [("Logistic Regression", y_proba_lr),
                    ("Gaussian Naive Bayes", y_proba_nb)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.show()

# 6.3 Comparative analysis table
def evaluate(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }

results = pd.DataFrame([
    evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    evaluate("Gaussian Naive Bayes", y_test, y_pred_nb, y_proba_nb),
]).set_index("Model").round(4)

print(results)

fig, ax = plt.subplots(figsize=(9, 4.5))
results.plot(kind="bar", ax=ax, rot=0)
ax.set_ylim(0.85, 1.005)
ax.set_ylabel("Score")
ax.set_title("Comparative Performance")
ax.legend(loc="lower right", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.show()

# 6.4 Cross-validation — confirms results are not an artefact of one split.
#     Scaling goes INSIDE the pipeline so it is re-fitted within each fold;
#     scaling the whole dataset first would leak information between folds.
for name, model in [("Logistic Regression",
                     LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
                    ("Gaussian NB", GaussianNB())]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"{name:<22} 5-fold ROC-AUC: "
          f"{scores.mean():.4f} (+/- {scores.std():.4f})")

# ---------------------------------------------------------------------------
# SECTION 7 — Probability concepts, made explicit
# ---------------------------------------------------------------------------
# 7.1 CONDITIONAL PROBABILITY
#     Precision and recall are conditional probabilities computed from the
#     confusion matrix. Verified here by hand against sklearn's values.
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = cm_lr.ravel()   # positive class = benign (label 1)

precision_manual = tp / (tp + fp)
recall_manual = tp / (tp + fn)
specificity_manual = tn / (tn + fp)

print("CONDITIONAL PROBABILITY")
print(f"  P(Benign | Predicted Benign)     = {tp}/{tp + fp} = {precision_manual:.4f}")
print(f"  P(Predicted Benign | Benign)     = {tp}/{tp + fn} = {recall_manual:.4f}")
print(f"  P(Predicted Malignant | Malignant) = {tn}/{tn + fp} = {specificity_manual:.4f}")
print(f"\n  sklearn precision = {precision_score(y_test, y_pred_lr):.4f}  (matches)")
print(f"  sklearn recall    = {recall_score(y_test, y_pred_lr):.4f}  (matches)")

# 7.2 MAXIMUM LIKELIHOOD ESTIMATION — the fitted coefficients
coefs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": log_reg.coef_[0],
})
coefs["abs"] = coefs["coefficient"].abs()
coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")

print("MAXIMUM LIKELIHOOD ESTIMATION")
print(f"  Intercept (beta_0): {log_reg.intercept_[0]:.4f}")
print(f"  Converged in {log_reg.n_iter_[0]} iterations")
print("\n  Top 10 coefficients by magnitude:")
print(coefs.head(10).to_string(index=False))

# 7.3 LOG-ODDS -> ODDS RATIOS
#     Logistic regression outputs log-odds; exponentiating gives odds ratios,
#     which are directly interpretable.
odds = coefs.head(8).copy()
odds["odds_ratio"] = np.exp(odds["coefficient"])
odds["effect"] = np.where(odds["coefficient"] > 0,
                          "increases odds of BENIGN",
                          "increases odds of MALIGNANT")

print("LOG-ODDS AND ODDS RATIOS (per 1 standard deviation increase)")
print(odds.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
top = coefs.head(12).sort_values("coefficient")
colors = ["#c0392b" if c < 0 else "#2874a6" for c in top["coefficient"]]
ax.barh(top["feature"], top["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Feature Importance — Logistic Regression\n"
             "blue = pushes toward benign, red = pushes toward malignant")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()

# 7.4 BAYESIAN INFERENCE — prior updated to posterior
prior_benign = y_train.mean()
prior_malignant = 1 - prior_benign

print("BAYESIAN INFERENCE")
print(f"  Prior P(Benign)    = {prior_benign:.4f}")
print(f"  Prior P(Malignant) = {prior_malignant:.4f}")
print(f"  (sklearn's stored priors: {gnb.class_prior_.round(4)})")

print("\n  Posterior probabilities after observing each patient's features:")
posterior = gnb.predict_proba(X_test_scaled)
sample_idx = [0, 1, 2, 3, 4]
for i in sample_idx:
    actual = "Benign" if y_test.iloc[i] == 1 else "Malignant"
    print(f"    Patient {i}: P(Malignant|x) = {posterior[i][0]:.4f}, "
          f"P(Benign|x) = {posterior[i][1]:.4f}   [actual: {actual}]")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(prior_malignant, color="grey", linestyle="--",
           label=f"Prior P(Malignant) = {prior_malignant:.3f}")
ax.scatter(range(len(sample_idx)), [posterior[i][0] for i in sample_idx],
           s=90, zorder=3, label="Posterior P(Malignant | features)")
ax.set_xticks(range(len(sample_idx)))
ax.set_xticklabels([f"Patient {i}" for i in sample_idx])
ax.set_ylabel("Probability of malignancy")
ax.set_title("Bayesian Updating: Prior vs Posterior")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/bayesian_updating.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# SECTION 8 — Clinical error analysis
# ---------------------------------------------------------------------------
# In cancer screening a false negative (malignant predicted benign) is far more
# costly than a false positive. This is the analysis paragraph in the report.
for name, preds in [("Logistic Regression", y_pred_lr),
                    ("Gaussian Naive Bayes", y_pred_nb)]:
    cm_ = confusion_matrix(y_test, preds)
    malignant_caught   = cm_[0, 0]   # actual malignant, predicted malignant
    malignant_missed   = cm_[0, 1]   # actual malignant, predicted benign  <-- costly
    benign_flagged     = cm_[1, 0]   # actual benign, predicted malignant
    benign_correct     = cm_[1, 1]

    total_malignant = malignant_caught + malignant_missed
    print(f"{name}")
    print(f"  Malignant cases MISSED (clinically costly): {malignant_missed}")
    print(f"  Benign cases wrongly flagged:               {benign_flagged}")
    print(f"  Malignant correctly identified:             "
          f"{malignant_caught}/{total_malignant} "
          f"({malignant_caught / total_malignant:.1%})\n")

print(results)