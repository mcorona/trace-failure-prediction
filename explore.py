from datasets import load_dataset

ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"]
print(ds.column_names)
print(ds[0]["mistake_step"], len(ds[0]["history"]))
