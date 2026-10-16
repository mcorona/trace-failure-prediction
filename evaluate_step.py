import re
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

MIN, B = 10, 300
ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]
ERR = re.compile(r"traceback|error|exception|failed", re.I)
EXIT = re.compile(r"exitcode: [1-9]")

def txt(m): return m["content"] or ""
def jacc(a, b):
    A, B_ = set(a.lower().split()), set(b.lower().split())
    return len(A & B_) / len(A | B_) if A | B_ else 0.0

# Features del paso actual (sin posición t ni longitud total)
rows = []
for gid, r in enumerate(ds):
    h, s = r["history"], int(r["mistake_step"])
    for k in range(1, len(h) + 1):
        m = h[k-1]; c = txt(m)
        prev = h[k-2] if k > 1 else None
        pc = txt(prev) if prev else ""
        recent = [txt(x) for x in h[max(0, k-3):k]]
        rows.append({
            "gid": gid, "t": k, "y": int(s <= k-1),
            "last_exit_fail": int(bool(EXIT.search(c))),
            "last_err": int(bool(ERR.search(c))),
            "last_is_terminal": int(m["name"] == "Computer_terminal"),
            "sim_prev": jacc(c, pc) if prev else 0.0,
            "same_agent_repeat": int(prev is not None and m["name"] == prev["name"]),
            "n_err_recent3": sum(bool(ERR.search(x)) for x in recent),
            "last_len": np.log1p(len(c)),
        })
df = pd.DataFrame(rows)
F = [c for c in df.columns if c not in ("gid", "t", "y")]
y, g = df["y"].values, df["gid"].values

# Folds idénticos a los de baseline (GroupKFold determinista)
df["fold"] = -1
for k, (_, te) in enumerate(GroupKFold(5).split(df, y, g)):
    df.loc[df.index[te], "fold"] = k

def score(d, col):
    res, w = [], []
    for _, s in d.groupby(["fold", "t"]):
        if min(s["y"].sum(), (s["y"] == 0).sum()) >= MIN:
            res.append(roc_auc_score(s["y"], s[col])); w.append(len(s))
    return np.average(res, weights=w) if res else np.nan

print("== AUROC univariado dentro de (fold, t) ==")
for f in F:
    print(f"{f:20s} {score(df, f):.3f}")

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gbm": GradientBoostingClassifier(random_state=0),
}
X = df[F].values
for name, m in models.items():
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    df[f"oof_{name}"] = oof

print("\n== Modelos dentro de (fold, t) ==")
groups = {gg: s for gg, s in df.groupby("gid")}
gids = np.array(list(groups))
rng = np.random.default_rng(0)
for name in models:
    col = f"oof_{name}"
    point = score(df, col)
    boot = []
    for _ in range(B):
        d = pd.concat([groups[x] for x in rng.choice(gids, len(gids), replace=True)])
        boot.append(score(d, col))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    print(f"{name:8s} point={point:.3f}  IC95%=[{lo:.3f}, {hi:.3f}]")

df.to_csv("step_features.csv", index=False)
