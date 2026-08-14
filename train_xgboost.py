import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load saved historical rumor records
df = pd.read_csv("historical_transfer_dataset.csv")

# X = Numerical features extracted by Llama; Y = Ground Truth (1 = Deal Landed, 0 = Failed)
X = df[["source_tier", "urgency_level", "mention_frequency"]]
y = df["deal_completed"]

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X, y)

# Save trained weight matrix to file
joblib.dump(model, "transfer_model.pkl")
