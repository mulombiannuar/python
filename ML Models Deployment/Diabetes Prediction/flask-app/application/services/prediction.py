import joblib
import pandas as pd
import os

# load the Random Forest model from the models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
model_path = os.path.join(models_dir, 'diabetes_random_forest_model.pkl')
loaded_rclf = joblib.load(model_path)

def make_prediction(data: dict) -> str:
    """
    Predicts whether a patient has diabetes using the Random Forest model.

    Parameters:
    - data (dict): A dictionary of input features, e.g.,
      {
        'pregnant': 2,
        'insulin': 130,
        'bmi': 28.1,
        'age': 45,
        'glucose': 150,
        'bp': 70,
        'pedigree': 0.5
      }

    Returns:
    - str: 'Has Diabetes' or 'No Diabetes'
    """
    input_df = pd.DataFrame([data])
    prediction = loaded_rclf.predict(input_df)[0]
    return "The Patient Has Diabetes" if prediction == 1 else "The Patient Does Not Have Diabetes"
