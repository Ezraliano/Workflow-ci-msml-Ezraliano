import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os
import sys
import warnings

# Suppress warning agar log bersih
warnings.filterwarnings("ignore")

# --- Ambil argumen dari CLI (MLflow akan inject ini via MLproject) ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./Crop_recommendation_prepocessing.csv")
args = parser.parse_args()

# --- Tracking URI untuk MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load dataset
try:
    df = pd.read_csv(args.dataset_path)
except FileNotFoundError:
    print(f"❌ Data tidak ditemukan di path: {args.dataset_path}")
    sys.exit(1)

# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inisialisasi model Logistic Regression
model = LogisticRegression(max_iter=200)

# Mulai run MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Training selesai")
    print("🔢 Akurasi:", acc)
    print(classification_report(y_test, y_pred))

    # Logging manual (opsional)
    mlflow.log_metric("accuracy_manual", acc)
    mlflow.sklearn.log_model(model, "model")
