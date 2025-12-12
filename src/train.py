import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate():
    print("Loading processed data...")
    df = pd.read_csv('data/processed_data.csv')
    
    # UPDATED Feature List matching preprocessing
    features = ['distance', 'surge_multiplier', 'cab_type', 'name', 'hour', 'source', 'destination']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    print("Training XGBoost Model... (this may take a moment)")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*30)
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ MAE: ${mae:.2f}")
    print("="*30 + "\n")
    
    # Save Model
    joblib.dump(model, 'dynamic_pricing_model.pkl')
    print("Model saved!")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.savefig('model_performance.png')
    print("Graph saved as model_performance.png")

if __name__ == "__main__":
    train_and_evaluate()