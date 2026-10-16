from datasets import load_dataset
from collections import Counter

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]

msg = ds[0]["history"][0]
print(type(msg), list(msg.keys()) if isinstance(msg, dict) else msg)

print("is_correct:", Counter(ds["is_correct"]))
print("agentes que fallan:", Counter(ds["mistake_agent"]).most_common(5))
print("longitud de historia:", Counter(len(h) for h in ds["history"]).most_common(5))

bad = [i for i, r in enumerate(ds)
       if r["mistake_step"] is None or not (0 <= r["mistake_step"] < len(r["history"]))]
print("trazas con mistake_step fuera de rango:", len(bad), bad[:10])
