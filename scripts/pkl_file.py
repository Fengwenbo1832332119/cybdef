import pickle, pandas as pd

path = r"C:\cybdef\datasets\cc2_causal_train.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

X, y = data["X"], data["y"]

print("✅ 数据维度:", X.shape)
print("🔹 特征预览:")
print(X.head())

print("\n🔹 标签分布:")
print(y.value_counts())
