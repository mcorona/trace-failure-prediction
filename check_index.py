from datasets import load_dataset

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]

steps = [int(v) for v in ds["mistake_step"]]
n = [len(h) for h in ds["history"]]
print("rango:", min(steps), max(steps))
print("fuera de rango base 0:", sum(not (0 <= s < L) for s, L in zip(steps, n)))
print("fuera de rango base 1:", sum(not (1 <= s <= L) for s, L in zip(steps, n)))

for i in range(3):
    r = ds[i]
    s = int(r["mistake_step"])
    print(f"\n--- traza {i} | mistake_step={s} | agente={r['mistake_agent']}")
    print("razón:", r["mistake_reason"][:250])
    print("history[s] base 0:", r["history"][s]["name"], "|", r["history"][s]["content"][:250])
    if 1 <= s <= len(r["history"]):
        print("history[s-1] base 1:", r["history"][s-1]["name"], "|", r["history"][s-1]["content"][:250])
