# AI-Powered Energy Consumption Forecasting System ⚡

An end-to-end machine learning pipeline that cleans industrial energy data and utilizes a Multi-Layer Perceptron (MLP) Neural Network to predict future power grid demand.

## 🚀 Key Features
- **Robust Data Cleaning:** Handles complex datetime formats (backslash/forward-slash) and missing values.
- **Feature Engineering:** Extracts temporal features (hour, day, month) to improve model accuracy.
- **Neural Network Architecture:** Implements `MLPRegressor` with standard scaling for high-performance regression.
- **Automated Visualization:** Generates performance graphs comparing actual vs. predicted consumption.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Joblib
- **Algorithm:** Multi-Layer Perceptron (MLP)

## 📊 Results
- **R2 Score:** 0.72 (Approx)
- **RMSE:** ~3417 MW
*(Insert your prediction_graph.png here later)*

## 📂 Project Structure
- `main.py`: The core execution script.
- `models/`: Contains the trained `.pkl` model and scaler.
- `outputs/`: Performance visualizations.

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/9cf35263-2ed6-4dd1-bb29-fe9f575d70cf" />

