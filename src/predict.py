import joblib
import pandas as pd
import numpy as np

def load_resources():
    # Load model and encoders
    print("Loading model resources...")
    model = joblib.load('dynamic_pricing_model.pkl')
    cab_type_le = joblib.load('data/cab_type_encoder.pkl')
    name_le = joblib.load('data/name_encoder.pkl')
    source_le = joblib.load('data/source_encoder.pkl')
    dest_le = joblib.load('data/destination_encoder.pkl')
    
    return model, cab_type_le, name_le, source_le, dest_le

def get_valid_input(prompt, encoder):
    """
    Helper function to handle case-insensitivity.
    If user types 'uber', it automatically finds 'Uber'.
    """
    valid_options = list(encoder.classes_)
    valid_options_lower = [x.lower() for x in valid_options]
    
    while True:
        print(f"\nOptions: {valid_options}")
        user_input = input(prompt).strip()
        
        # Try to find a match (ignoring case)
        if user_input.lower() in valid_options_lower:
            # Find the original case-sensitive name
            index = valid_options_lower.index(user_input.lower())
            correct_name = valid_options[index]
            return encoder.transform([correct_name])[0]
        else:
            print(f"❌ '{user_input}' is not valid. Please try again.")

def get_user_input(cab_le, name_le, src_le, dest_le):
    print("\n" + "="*40)
    print(" 🚗  RIDE PRICE PREDICTOR ")
    print("="*40)
    
    # Simple Numeric Inputs
    try:
        distance = float(input("\n1. Enter Distance (miles): "))
        surge = float(input("2. Enter Surge Multiplier (1.0 = normal, 2.0 = double): "))
        hour = int(input("3. Enter Hour of Day (0-23): "))
    except ValueError:
        print("❌ Error: Please enter numbers only for distance, surge, and hour.")
        return None

    # Smart Text Inputs (Handles capitalization automatically)
    print("\n--- Ride Details ---")
    cab_idx = get_valid_input("4. Enter Cab Type (e.g., Uber): ", cab_le)
    name_idx = get_valid_input("5. Enter Ride Name (e.g., UberXL): ", name_le)
    
    print("\n--- Location ---")
    src_idx = get_valid_input("6. Enter Source (e.g., North Station): ", src_le)
    dest_idx = get_valid_input("7. Enter Destination: ", dest_le)
    
    # Create DataFrame
    input_data = pd.DataFrame({
        'distance': [distance],
        'surge_multiplier': [surge],
        'cab_type': [cab_idx],
        'name': [name_idx],
        'hour': [hour],
        'source': [src_idx],
        'destination': [dest_idx]
    })
    
    return input_data

def predict_price():
    try:
        model, cab_le, name_le, src_le, dest_le = load_resources()
        input_data = get_user_input(cab_le, name_le, src_le, dest_le)
        
        if input_data is not None:
            predicted_price = model.predict(input_data)[0]
            
            print("\n" + "*"*50)
            print(f"💰 ESTIMATED PRICE: ${predicted_price:.2f}")
            print("*"*50 + "\n")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    predict_price()