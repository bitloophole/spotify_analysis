import pandas as pd
from pathlib import Path

in_path = Path("spark-apps/data/spotify_songs.csv")
df = pd.read_csv(in_path)

# optional: sample smaller portion if dataset is huge
# df = df.sample(frac=0.3, random_state=42)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
n = len(df)
a = n // 3
b = 2 * n // 3

out_master = Path("spark-data/master/spotify_master.csv")
out_w1 = Path("spark-data/worker1/spotify_w1.csv")
out_w2 = Path("spark-data/worker2/spotify_w2.csv")

out_master.parent.mkdir(parents=True, exist_ok=True)
out_w1.parent.mkdir(parents=True, exist_ok=True)
out_w2.parent.mkdir(parents=True, exist_ok=True)

df.iloc[:a].to_csv(out_master, index=False)
df.iloc[a:b].to_csv(out_w1, index=False)
df.iloc[b:].to_csv(out_w2, index=False)

print("Wrote:", out_master, out_w1, out_w2)
print("Counts:", len(df.iloc[:a]), len(df.iloc[a:b]), len(df.iloc[b:]))

