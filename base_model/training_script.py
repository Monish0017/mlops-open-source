import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load train data
train = pd.read_csv("../dataset/train.csv")
X, y = train.drop("label", axis=1), train["label"]

# Train model (example)
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
