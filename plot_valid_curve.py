import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/yechenlu/KuDA/checkpoints/SIMS/coldstart/valid_curve.csv"

assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"

df = pd.read_csv(csv_path)

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