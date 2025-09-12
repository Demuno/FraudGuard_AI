import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = os.path.join(".", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detection_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
DATA_PATH = os.path.join(".", "data", "transactions.csv")
MAX_SIZE_MB = 100
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

# Ajustando o tamanho do arquivo para subir ao GitHub

if not os.path.exists(DATA_PATH):
    print(f"Error: File not found: {DATA_PATH}.")
else:
    current_size = os.path.getsize(DATA_PATH)
    print(f"Size: {current_size / (1024 * 1024):.2f} MB")

    if current_size <= MAX_SIZE_BYTES:
        print(f"The file size is within the limit of {MAX_SIZE_MB} MB. No action needed.")
    else:
        print(f"The file size is above the limit of {MAX_SIZE_MB} MB. Reducing file size...")
        try:
            df_full = pd.read_csv(DATA_PATH, sep=',', on_bad_lines='skip')
            total_rows = len(df_full)

            if total_rows == 0:
                print("The file is empty after reading. No data to process.")
            else:
                avg_bytes_per_row = current_size / total_rows   
                target_rows = int((MAX_SIZE_BYTES / avg_bytes_per_row) * 0.95)
                target_rows = max(1, min(target_rows, total_rows))

                if target_rows == total_rows:
                    print("The file size is still above the limit, but cannot be reduced further without losing all data.")
                else:
                    df_reduced = df_full.head(target_rows)
                    df_reduced.to_csv(DATA_PATH, index=False, sep=',')
                    
                    new_size = os.path.getsize(DATA_PATH)
                    print(f"File size reduced from {len(df_reduced)} lines. New size: {new_size / (1024 * 1024):.2f} MB")

                    if new_size > MAX_SIZE_BYTES:
                        print(f"The file size is still above the limit of {MAX_SIZE_MB}")
                    else:
                        print("File size is now within the limit.")

        except Exception as e:
            print(f"An error occurred: {e}")

features = [col for col in df_full.columns if col not in ['Time', 'Class']]

x = df_full[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print("Features scaled.")

model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
model.fit(x_scaled)

print("Model trained.")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"The model was saved in: {MODEL_PATH}")
print(f"The scaler was saved in: {SCALER_PATH}")
