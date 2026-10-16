import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

B = 300
df = pd.read_csv("step_features.csv")

s_idx = df[df["y"] == 1].groupby("gid")["t"].min() - 1
df["s"] = df["gid"].map(s_idx)
df["y_ev"] = (df["t"] - 1 == df["s"]).astype(int)
df["y_win"] = ((df["t"] - 1 >= df["s"]) & (df["t"] - 1 <= df["s"] + 1)).astype(int)

TCOLS = []
for v in sorted(df["t"].unique()):
    df[f"T{v}"] = (df["t"] == v).astype(int)
    TCOLS.append(f"T{v}")

F = ["last_exit_fail", "last_err", "last_is_terminal", "sim_prev",
     "same_agent_repeat", "n_err_recent3", "last_len"]
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
gb = lambda: GradientBoostingClassifier(random_state=0)

models = {
    "onehot_t (control)": (TCOLS, lr),
    "gbm_t (control)": (["t"], gb),
    "logreg_step": (F, lr),
    "gbm_step": (F, gb),
    "gbm_step+t": (F + ["t"], gb),
}
CONTROL = "onehot_t (control)"

gids = df["gid"].unique()
idx = {g: np.where(df["gid"].values == g)[0] for g in gids}
rng = np.random.default_rng(0)

def oof(label):
    out = {}
    for name, (cols, mk) in models.items():
        p = np.zeros(len(df))
        for k in sorted(df["fold"].unique()):
            tr = (df["fold"] != k).values
            m = mk()
            m.fit(df.loc[tr, cols].values, df.loc[tr, label].values)
            p[~tr] = m.predict_proba(df.loc[~tr, cols].values)[:, 1]
        out[name] = p
    return out

for label in ["y_ev", "y_win"]:
    y = df[label].values
    print(f"\n== etiqueta {label}: positivos={y.sum()} ({y.mean():.1%}) ==")
    preds = oof(label)
    for name in models:
        print(f"{name:20s} AUROC pooled={roc_auc_score(y, preds[name]):.3f}")

    boot = {n: [] for n in models}
    for _ in range(B):
        sel = np.concatenate([idx[g] for g in rng.choice(gids, len(gids), replace=True)])
        if len(np.unique(y[sel])) < 2:
            continue
        for n in models:
            boot[n].append(roc_auc_score(y[sel], preds[n][sel]))

    print("-- IC95% de AUROC --")
    for n in models:
        lo, hi = np.percentile(boot[n], [2.5, 97.5])
        print(f"{n:20s} [{lo:.3f}, {hi:.3f}]")

    print("-- margen vs onehot_t (control principal) --")
    rng2 = np.random.default_rng(1)
    for n in models:
        if n == CONTROL:
            continue
        diffs = []
        for _ in range(B):
            sel = np.concatenate([idx[g] for g in rng2.choice(gids, len(gids), replace=True)])
            if len(np.unique(y[sel])) < 2:
                continue
            diffs.append(roc_auc_score(y[sel], preds[n][sel]) - roc_auc_score(y[sel], preds[CONTROL][sel]))
        point = roc_auc_score(y, preds[n]) - roc_auc_score(y, preds[CONTROL])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"{n:20s} punto={point:+.3f}  IC95%=[{lo:+.3f}, {hi:+.3f}]")
