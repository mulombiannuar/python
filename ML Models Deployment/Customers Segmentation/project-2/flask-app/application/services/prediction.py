import joblib
import pandas as pd
import os

# load the KMeans model from the models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

model_path = os.path.join(models_dir, 'kmeans_model.pkl')
loaded_kmeans = joblib.load(model_path)

scalar_path = os.path.join(models_dir, 'scaler.pkl')
loaded_scaler = joblib.load(scalar_path)

def map_cluster_to_label(cluster_number: int) -> str:
    """
    Maps a cluster number to a descriptive label.

    Args:
        cluster_number (int): The cluster number.

    Returns:
        str: The descriptive label for the cluster.
    """
    cluster_labels = {
        0: "Low-Spend Active Customers",
        1: "High-Value Engaged Customers",
        2: "Affluent but Unengaged Customers",
        3: "Disengaged Low Spenders",
        4: "Online Shoppers with Low Loyalty"
    }
    return cluster_labels.get(cluster_number, "Unknown Cluster")
  

def make_prediction(input_data: dict) -> str:
    """
    Predicts the customer cluster and its corresponding descriptive label 
    based on input features using a pre-trained KMeans model.

    Parameters:
    - input_data (dict): A dictionary with input features.
      Expected keys:
        - 'Income': Annual income (numeric)
        - 'Age': Customer age (numeric)
        - 'Total_Spending': Total amount spent (numeric)
        - 'Recency': Days since last purchase (numeric)
        - 'NumWebPurchases': Number of website purchases (numeric)
        - 'NumStorePurchases': Number of in-store purchases (numeric)
        - 'AcceptedAny': Whether accepted any marketing campaign (binary: 0 or 1)
        - 'NumWebVisitsMonth': Website visits per month (numeric)

      Example:
      ```python
      input_data = {
          'Income': 50000,
          'Age': 45,
          'Total_Spending': 600,
          'Recency': 30,
          'NumWebPurchases': 5,
          'NumStorePurchases': 4,
          'AcceptedAny': 1,
          'NumWebVisitsMonth': 6
      }
      ```

    Returns:
    - str: A string indicating the predicted cluster number and its descriptive label.
      Example: "Customer belongs to Cluster 2 - Affluent but Unengaged Customers"
    """
    
    # convert the input dictionary to a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # scale the input
    scaled_input = loaded_scaler.transform(input_df)

    # predict the cluster
    cluster_number = loaded_kmeans.predict(scaled_input)[0]

    # map to descriptive label
    cluster_label = map_cluster_to_label(cluster_number)

    return f"Customer belongs to Cluster {cluster_number} - {cluster_label}"

