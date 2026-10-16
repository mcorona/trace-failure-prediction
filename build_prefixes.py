import re, csv
from statistics import mean
from datasets import load_dataset

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]

ERR = re.compile(r"traceback|exitcode: [1-9]|error|exception|failed", re.I)

def jacc(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / len(A | B) if A | B else 0.0

rows = []
mismatch = 0
for gid, r in enumerate(ds):
    h, s = r["history"], int(r["mistake_step"])
    if h[s]["name"] != r["mistake_agent"]:
        mismatch += 1
    for k in range(1, len(h) + 1):
        pref = h[:k]
        cs = [m["content"] or "" for m in pref]
        errs = [bool(ERR.search(c)) for c in cs]
        rows.append({
            "gid": gid, "trace_id": r["question_ID"], "t": k,
            "y": int(s <= k - 1),                # error decisivo ya en el prefijo
            "n_msgs": k,
            "n_err": sum(errs),
            "err_ratio": sum(errs) / k,
            "last_err": int(errs[-1]),
            "mean_sim": mean(jacc(cs[i-1], cs[i]) for i in range(1, k)) if k > 1 else 0.0,
            "n_switches": sum(pref[i]["name"] != pref[i-1]["name"] for i in range(1, k)),
            "mean_len": mean(len(c) for c in cs),
            "n_total_ANALISIS_ONLY": len(h),     # NO usar como feature
        })

with open("prefixes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print("trazas con agente mismatch:", mismatch)
print("filas:", len(rows), "| positivos:", sum(r["y"] for r in rows))
