import joblib
import pandas as pd
import os

# load the model from the models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

model_path = os.path.join(models_dir, 'final_best_model.pkl')
loaded_model = joblib.load(model_path)

scalar_path = os.path.join(models_dir, 'scaler.pkl')
loaded_scaler = joblib.load(scalar_path)

dummy_columns_path = os.path.join(models_dir, 'dummy_columns.pkl')
loaded_dummy_columns = joblib.load(dummy_columns_path)

def make_prediction(data: dict) -> str:
    """
    Predicts the cluster for a given customer input using the KMeans model.
    """

    # Convert to DataFrame
    user_input_df = pd.DataFrame([data])
    
    # Define Ordinal Mappings
    diet_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    parental_education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}
    internet_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}

    user_input_df['diet_quality_e'] = user_input_df['diet_quality'].map(diet_quality_map)
    user_input_df['parental_education_level_e'] = user_input_df['parental_education_level'].map(parental_education_map)
    user_input_df['internet_quality_e'] = user_input_df['internet_quality'].map(internet_quality_map)

    # One-Hot Encode Nominal Variables
    dummies = pd.get_dummies(user_input_df[['gender', 'part_time_job', 'extracurricular_participation']], drop_first=True)
    dummies = dummies.reindex(columns=loaded_dummy_columns, fill_value=0)

    # Drop original categorical columns
    user_input_df = user_input_df.drop([
        'gender',
        'part_time_job',
        'diet_quality',
        'parental_education_level',
        'internet_quality',
        'extracurricular_participation'
    ], axis=1)

    # Concatenate numeric and encoded categorical features
    final_input = pd.concat([user_input_df, dummies], axis=1)

    # Ensure columns match model expectation
    final_input = final_input.reindex(columns=loaded_scaler.feature_names_in_, fill_value=0)

    # Scale the input
    scaled_input = loaded_scaler.transform(final_input)

    # Predict
    predicted_score = loaded_model.predict(scaled_input)[0]
    return f"Predicted Exam Score: {predicted_score:.2f}"
