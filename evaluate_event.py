import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

B = 300
df = pd.read_csv("step_features.csv")

# Paso de error por traza: primer t con y=1, menos 1 (índice base 0)
s_idx = df[df["y"] == 1].groupby("gid")["t"].min() - 1
df["s"] = df["gid"].map(s_idx)
df["y_ev"] = (df["t"] - 1 == df["s"]).astype(int)
df["y_win"] = ((df["t"] - 1 >= df["s"]) & (df["t"] - 1 <= df["s"] + 1)).astype(int)

F = ["last_exit_fail", "last_err", "last_is_terminal", "sim_prev",
     "same_agent_repeat", "n_err_recent3", "last_len"]
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
models = {
    "control_t":   (["t"], lr),
    "logreg_step": (F, lr),
    "gbm_step":    (F, lambda: GradientBoostingClassifier(random_state=0)),
}

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
    print(f"\n== etiqueta {label}: positivos={y.sum()} de {len(y)} ({y.mean():.1%}) ==")
    preds = oof(label)
    for name in models:
        print(f"{name:12s} AUROC pooled={roc_auc_score(y, preds[name]):.3f}")
    boot = {n: [] for n in models}
    margin = {"logreg_step - control": [], "gbm_step - control": []}
    for _ in range(B):
        sel = np.concatenate([idx[g] for g in rng.choice(gids, len(gids), replace=True)])
        if len(np.unique(y[sel])) < 2:
            continue
        a = {n: roc_auc_score(y[sel], preds[n][sel]) for n in models}
        for n in models:
            boot[n].append(a[n])
        margin["logreg_step - control"].append(a["logreg_step"] - a["control_t"])
        margin["gbm_step - control"].append(a["gbm_step"] - a["control_t"])
    for n in models:
        lo, hi = np.percentile(boot[n], [2.5, 97.5])
        print(f"{n:12s} IC95%=[{lo:.3f}, {hi:.3f}]")
    for k, v in margin.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"margen {k:24s} IC95%=[{lo:.3f}, {hi:.3f}]")
