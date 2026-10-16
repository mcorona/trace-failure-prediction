from datasets import load_dataset

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]

for gid, r in enumerate(ds):
    h, s = r["history"], int(r["mistake_step"])
    if h[s]["name"] != r["mistake_agent"]:
        print(f"\n=== gid={gid} | step={s} | marcado={r['mistake_agent']}")
        print("history[s]   :", h[s]["name"], "|", (h[s]["content"] or "")[:150])
        print("history[s-1] :", h[s-1]["name"] if s > 0 else "-")
        print("agentes en traza:", sorted({m["name"] for m in h}))
        print("razón:", r["mistake_reason"][:200])
