import numpy as np, pandas as pd

np.random.seed(42)
n = 100

df = pd.DataFrame({
    "age": np.random.randint(29, 78, n),
    "gender": np.random.choice(["male", "female"], n),
    "cp": np.random.randint(0, 4, n),
    "trestbps": np.random.randint(94, 201, n),
    "chol": np.random.randint(126, 565, n),
    "fbs": np.random.randint(0, 2, n),
    "restecg": np.random.randint(0, 3, n),
    "thalach": np.random.randint(71, 203, n),
    "exang": np.random.randint(0, 2, n),
    "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),
    "slope": np.random.randint(0, 3, n),
    "ca": np.random.randint(0, 5, n),
    "thal": np.random.randint(0, 4, n),
})
df.to_csv("synthetic_100.csv", index=False)
print("Saved synthetic_100.csv")
