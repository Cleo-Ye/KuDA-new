import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/yechenlu/KuDA/checkpoints/SIMS/valid_curve.csv"

assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"

# 兼容混合格式：旧 run 写 7 列，新 run 写 9 列，取最后一次出现的表头及其后的数据
with open(csv_path, encoding="utf-8") as f:
    lines = f.readlines()
header_idx = None
for i in range(len(lines) - 1, -1, -1):
    if lines[i].strip().startswith("epoch"):
        header_idx = i
        break
if header_idx is None:
    raise ValueError("No header line found in CSV")
header = [c.strip() for c in lines[header_idx].strip().split(",")]
n_cols = len(header)
data_rows = []
for i in range(header_idx + 1, len(lines)):
    parts = [p.strip() for p in lines[i].strip().split(",")]
    if len(parts) == n_cols:
        data_rows.append(parts)
df = pd.DataFrame(data_rows, columns=header)
for c in header:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["epoch"])
df["epoch"] = df["epoch"].astype(int)

# 基本检查：只打印前 5 行确认读对文件（可按需注释掉）
print(df.head())

epochs = df["epoch"]

plt.figure(figsize=(10, 5))
plt.plot(epochs, df["train_mae"], label="train MAE", color="tab:blue")
plt.plot(epochs, df["valid_mae"], label="valid MAE", color="tab:orange")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Train / Valid MAE vs Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mae_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, df["train_corr"], label="train Corr", color="tab:green")
plt.plot(epochs, df["valid_corr"], label="valid Corr", color="tab:red")
plt.xlabel("Epoch")
plt.ylabel("Pearson Corr")
plt.title("Train / Valid Corr vs Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("corr_curve.png", dpi=200)
plt.close()