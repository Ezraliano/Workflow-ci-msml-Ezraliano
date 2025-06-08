import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import sys
import warnings

# Suppress warning agar log bersih
warnings.filterwarnings("ignore")

# --- Path relatif terhadap posisi file ini dijalankan di CI ---
DATA_PATH = os.getenv("DATA_PATH", "./Crop_recommendation_prepocessing.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Set URI tracking MLflow (harus bisa diakses di lingkungan CI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Autolog model dan metrik ke MLflow
mlflow.sklearn.autolog()

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ Data tidak ditemukan di path: {DATA_PATH}")
    sys.exit(1)

# Pisah fitur dan target
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inisialisasi model
model = LogisticRegression(max_iter=200)

# Mulai MLflow run
with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Training selesai")
    print("🔢 Akurasi:", acc)
    print(classification_report(y_test, y_pred))

    # Logging eksplisit (opsional karena autolog sudah mencatat)
    mlflow.log_metric("accuracy_manual", acc)
    mlflow.sklearn.log_model(model, "model")
