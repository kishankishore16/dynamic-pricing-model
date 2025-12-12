import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model():
    print("Loading data and model...")
    
    # 1. Load Data
    df = pd.read_csv('data/processed_data.csv')
    
    # Features used (Must match training exactly)
    features = ['distance', 'surge_multiplier', 'cab_type', 'name', 'hour', 'source', 'destination']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    # 2. Split Data (Same random_state=42 ensures we test on the same data as before)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Load Model
    model = joblib.load('dynamic_pricing_model.pkl')
    
    # 4. Predict
    print("Calculating metrics...")
    predictions = model.predict(X_test)
    
    # 5. Calculate Metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Calculate "MAPE" (Mean Absolute Percentage Error) - easier to understand
    # We add 0.001 to avoid dividing by zero
    mape = np.mean(np.abs((y_test - predictions) / (y_test + 0.001))) * 100
    
    # 6. Print Report
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANCE REPORT")
    print("="*40)
    
    print(f"1. Accuracy (R² Score):  {r2:.4f}")
    print("   (0.0 = Bad, 1.0 = Perfect. Anything above 0.90 is excellent)")
    
    print(f"\n2. Average Error (MAE):  ${mae:.2f}")
    print("   (On average, the prediction is off by this amount)")
    
    print(f"\n3. Root Mean Sq Error:   ${rmse:.2f}")
    print("   (Penalizes large mistakes more heavily)")
    
    print(f"\n4. Percentage Error:     {mape:.2f}%")
    print("   (On average, the price is off by this percentage)")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()