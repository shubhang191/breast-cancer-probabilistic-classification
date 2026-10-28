"""Reproducibility check for the held-out baseline, independent of notebooks."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def test_baseline_scores():
    ds = load_breast_cancer()
    xtr, xte, ytr, yte = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42, stratify=ds.target)
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    lr = LogisticRegression(max_iter=5000, random_state=42).fit(xtr, ytr)
    gnb = GaussianNB().fit(xtr, ytr)
    assert len(ytr) == 455 and len(yte) == 114
    assert accuracy_score(yte, lr.predict(xte)) == 112/114
    assert accuracy_score(yte, gnb.predict(xte)) == 106/114

if __name__ == '__main__':
    test_baseline_scores()
    print('Baseline reproducibility check passed')
