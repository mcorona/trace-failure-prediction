from datasets import load_dataset
from collections import Counter

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]

types = Counter(type(v).__name__ for v in ds["mistake_step"])
print("tipos:", types)

strs = [v for v in ds["mistake_step"] if isinstance(v, str)]
print("ejemplos de texto:", strs[:15])
print("valores no numéricos:", [v for v in strs if not v.strip().lstrip("-").isdigit()][:15])
