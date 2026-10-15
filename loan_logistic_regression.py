"""Loan eligibility with Logistic Regression.
x-axis = civil (CIBIL) score, y-axis = loan amount required.  Target: eligible (1) / not eligible (0).
Data is SYNTHETIC unless you pass your own CSV with columns: civil_score, loan_amount, eligible.
Run:  python loan_logistic_regression.py [your_data.csv]
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1) DATA ---------------------------------------------------------------
if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
else:
    rng = np.random.default_rng(42)
    n = 600
    score = np.clip(rng.normal(680, 90, n), 300, 900).round()
    amount = np.clip(rng.lognormal(np.log(500_000), 0.7, n), 50_000, 3_000_000).round(-3)
    z = 0.035 * (score - 650) - 0.0000035 * (amount - 700_000)      # hidden "bank rule"
    eligible = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)  # noisy decisions
    df = pd.DataFrame({"civil_score": score, "loan_amount": amount, "eligible": eligible})
    df.to_csv("loan_data_synthetic.csv", index=False)
print(df.describe().T[["min", "mean", "max"]]); print(df.eligible.value_counts())

# 2) TRAIN / TEST -------------------------------------------------------
X, y = df[["civil_score", "loan_amount"]], df["eligible"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression())      # scaling: score and amount are on very different scales
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(f"\nTest accuracy: {accuracy_score(yte, pred):.3f}")
print("Confusion matrix [rows=actual, cols=predicted]:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["Not eligible", "Eligible"]))

# 3) GRAPH: x = civil score, y = loan amount, with decision boundary ----
xx, yy = np.meshgrid(np.linspace(300, 900, 300), np.linspace(df.loan_amount.min(), df.loan_amount.max(), 300))
grid = pd.DataFrame({"civil_score": xx.ravel(), "loan_amount": yy.ravel()})
prob = model.predict_proba(grid)[:, 1].reshape(xx.shape)
fig, ax = plt.subplots(figsize=(9, 6))
cs = ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.35)
ax.contour(xx, yy, prob, levels=[0.5], colors="k", linewidths=2)
ax.scatter(df.civil_score[df.eligible == 1], df.loan_amount[df.eligible == 1], c="green", s=18, label="Eligible")
ax.scatter(df.civil_score[df.eligible == 0], df.loan_amount[df.eligible == 0], c="red", s=18, label="Not eligible")
ax.plot([], [], "k-", label="Decision boundary (p = 0.5)")
fig.colorbar(cs, label="Probability of eligibility")
ax.set_xlabel("Civil (CIBIL) score"); ax.set_ylabel("Loan amount required")
ax.set_title("Loan eligibility - Logistic Regression"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("loan_eligibility_plot.png", dpi=130)

# 4) PREDICT FOR ONE PERSON ---------------------------------------------
def check(score, amount):
    p = model.predict_proba(pd.DataFrame({"civil_score": [score], "loan_amount": [amount]}))[0, 1]
    return ("ELIGIBLE" if p >= 0.5 else "NOT ELIGIBLE"), p
for s, a in [(780, 400_000), (620, 900_000), (700, 1_500_000)]:
    r, p = check(s, a); print(f"score={s}, amount={a:,}: {r} (probability {p:.0%})")
