import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess(filepath):
    print(f"Loading data from {filepath}...")
    # Load data (using all rows since your file is manageable, or limit with nrows=100000)
    df = pd.read_csv(filepath)
    
    # 1. Handle Missing Prices
    df = df.dropna(subset=['price'])
    
    # 2. Create 'hour' from 'time_stamp' (Your file has ms timestamps)
    # 1544952607890 -> Datetime -> Hour
    df['datetime'] = pd.to_datetime(df['time_stamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    
    # 3. Select Features
    # We use Source/Dest because location changes price
    features = ['distance', 'surge_multiplier', 'cab_type', 'name', 'hour', 'source', 'destination']
    target = 'price'
    
    df = df[features + [target]]
    
    # 4. Encode Categorical Data (Text -> Numbers)
    le = LabelEncoder()
    categorical_cols = ['cab_type', 'name', 'source', 'destination']
    
    print("Encoding categorical data...")
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        # Save encoders for the prediction script
        joblib.dump(le, f'data/{col}_encoder.pkl')
        
    print("Data processed successfully!")
    print(df.head())
    return df

if __name__ == "__main__":
    try:
        df = load_and_preprocess('data/rides.csv')
        df.to_csv('data/processed_data.csv', index=False)
        print("✅ Saved to data/processed_data.csv")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")