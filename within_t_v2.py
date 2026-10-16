import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

MIN = 20
df = pd.read_csv("oof_predictions.csv")
df["bin"] = pd.cut(df["t"], bins=[0, 2, 4, 6, 8, 10, 100])

valid = []
for b, sub in df.groupby("bin", observed=True):
    pos, neg = int(sub["y"].sum()), int((sub["y"] == 0).sum())
    ok = min(pos, neg) >= MIN
    print(f"{str(b):10s} pos={pos:4d} neg={neg:4d} {'USA' if ok else 'EXCLUIDO'}")
    if ok:
        valid.append(sub)

vdf = pd.concat(valid)
for col in ["oof_trivial_n_msgs", "oof_logreg", "oof_gbm"]:
    res = [roc_auc_score(s["y"], s[col]) for s in valid]
    w = [len(s) for s in valid]
    print(f"{col:22s} within-t AUROC={np.average(res, weights=w):.3f}  bins={len(valid)}")
