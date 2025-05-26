import joblib
import pandas as pd
import os

# load the KMeans model from the models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

model_path = os.path.join(models_dir, 'kmeans_annual_income_model.pkl')
loaded_kmeans = joblib.load(model_path)

scalar_path = os.path.join(models_dir, 'scaler_annual_income.pkl')
loaded_scaler = joblib.load(scalar_path)

def make_prediction(data: dict) -> str:
    """
    Predicts the cluster for a given customer input using the KMeans model.

    Parameters:
    - data (dict): A dictionary of input features, e.g.,
      {
        'annual_income_k': 60,
        'spending_score': 30
      }

    Returns:
    - str: The predicted cluster label.
    """
    input_df = pd.DataFrame([data])
    scaled_input = loaded_scaler.transform(input_df)
    prediction = loaded_kmeans.predict(scaled_input)[0]
    return f"Customer belongs to Cluster {prediction}"
   
