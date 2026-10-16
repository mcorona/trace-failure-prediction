import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

MIN, B = 20, 300
cols = ["oof_trivial_n_msgs", "oof_logreg", "oof_gbm"]
df = pd.read_csv("oof_predictions.csv")

def score(d, col):
    res, w = [], []
    for _, s in d.groupby("t"):
        if min(s["y"].sum(), (s["y"] == 0).sum()) >= MIN:
            res.append(roc_auc_score(s["y"], s[col])); w.append(len(s))
    return np.average(res, weights=w) if res else np.nan

point = {c: score(df, c) for c in cols}
for c in cols:
    print(f"{c:22s} point={point[c]:.3f}")

groups = {g: s for g, s in df.groupby("gid")}
gids = np.array(list(groups))
rng = np.random.default_rng(0)
boot = {c: [] for c in cols}
margin = []
for _ in range(B):
    d = pd.concat([groups[g] for g in rng.choice(gids, len(gids), replace=True)])
    r = {c: score(d, c) for c in cols}
    for c in cols:
        boot[c].append(r[c])
    margin.append(r["oof_gbm"] - r["oof_trivial_n_msgs"])

for c in cols:
    lo, hi = np.nanpercentile(boot[c], [2.5, 97.5])
    print(f"{c:22s} IC95%=[{lo:.3f}, {hi:.3f}]")
lo, hi = np.nanpercentile(margin, [2.5, 97.5])
print(f"margen gbm - control   IC95%=[{lo:.3f}, {hi:.3f}]")
