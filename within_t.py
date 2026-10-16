import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("oof_predictions.csv")
df["bin"] = pd.cut(df["t"], bins=[0, 2, 4, 6, 8, 10, 100])

for col in ["oof_trivial_n_msgs", "oof_logreg", "oof_gbm"]:
    res, w = [], []
    for _, sub in df.groupby("bin", observed=True):
        if sub["y"].nunique() == 2:
            res.append(roc_auc_score(sub["y"], sub[col]))
            w.append(len(sub))
    print(f"{col:22s} within-t AUROC={np.average(res, weights=w):.3f}  bins={len(res)}")

print("\nprevalencia y por bin:")
print(df.groupby("bin", observed=True)["y"].agg(["mean", "size"]).round(2))
