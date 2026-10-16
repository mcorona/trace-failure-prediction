import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("prefixes.csv")
FULL = ["n_msgs", "n_err", "err_ratio", "last_err", "mean_sim", "n_switches", "mean_len"]
y, g = df["y"].values, df["gid"].values

models = {
    "trivial_n_msgs": (["n_msgs"], make_pipeline(StandardScaler(), LogisticRegression())),
    "logreg": (FULL, make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    "gbm": (FULL, GradientBoostingClassifier(random_state=0)),
}

for name, (cols, m) in models.items():
    X = df[cols].values
    oof, aucs = np.zeros(len(y)), []
    for tr, te in GroupKFold(5).split(X, y, g):
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
    print(f"{name:15s} AUROC oof={roc_auc_score(y, oof):.3f}  fold={np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    df[f"oof_{name}"] = oof

df.to_csv("oof_predictions.csv", index=False)
