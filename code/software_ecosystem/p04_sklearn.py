import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# synthetic "pump failure" data: rare positives (about 10%)
rng = np.random.default_rng(1)
n = 4000
X = pd.DataFrame({
    "vibration_rms": rng.gamma(2, 1, n),
    "temp_c": rng.normal(60, 8, n),
    "age_years": rng.uniform(0, 20, n),
    "site": rng.choice(["A", "B", "C"], n),
})
logit = -6 + 0.9 * X.vibration_rms + 0.05 * (X.temp_c - 60) + 0.1 * X.age_years + 0.8 * (X.site == "C")
y = rng.random(n) < 1 / (1 + np.exp(-logit))
print(f"positive rate: {y.mean():.3f}")

pre = ColumnTransformer([
    ("num", StandardScaler(), ["vibration_rms", "temp_c", "age_years"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
])
models = {
    "logistic": make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight="balanced")),
    "boosting": make_pipeline(pre, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)   # preprocessing is re-fitted inside each fold: no leakage
for name, model in models.items():
    ap = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    print(f"{name:9s} PR-AUC = {ap.mean():.3f} +/- {ap.std():.3f}   (a random guess scores {y.mean():.3f})")
