import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_PATH = "data/energy.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

def setup():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    setup()

    # 1. Load & Clean (Standard)
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found!")
        return

    data = pd.read_csv(DATA_PATH)
    data = data[["Datetime", "PJME_MW"]]
    data.columns = ["Datetime", "Energy"]
    data["Datetime"] = data["Datetime"].astype(str).str.replace("\\", "/", regex=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
    data = data.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    data = data.resample("h").mean().ffill()

    # 2. Optimization: Use a smaller sample if the dataset is huge
    if len(data) > 20000:
        print(f"💡 Large dataset detected ({len(data)} rows). Sampling last 20,000 rows for speed...")
        data = data.tail(20000)

    # 3. Features
    data["hour"] = data.index.hour
    data["day"] = data.index.dayofweek
    data["month"] = data.index.month

    X = data[["hour", "day", "month"]]
    y = data["Energy"]

    # 4. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Optimized Training (FASTER)
    # hidden_layer_sizes reduced slightly, max_iter set to 100 for quick results
    print("Training Model (Optimized for Speed)... ⏳")
    model = MLPRegressor(
        hidden_layer_sizes=(32, 32), 
        max_iter=100, 
        tol=1e-3,
        random_state=42,
        verbose=True # This shows you progress in the terminal!
    )
    
    model.fit(X_train_scaled, y_train)
    print("\nModel Training Completed ✅")

    # 6. Results
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"📊 Results -> RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")

    # 7. Save
    joblib.dump(model, f"{MODEL_DIR}/energy_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    # 8. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual", alpha=0.8)
    plt.plot(predictions[:100], label="Predicted", linestyle="--")
    plt.title("Energy Forecast (Fast Mode)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/prediction_graph.png")
    plt.close()
    print("Everything Saved! Check 'models/' and 'outputs/' folders. ✅")

if __name__ == "__main__":
    main()