import os
import joblib

# directory where models are stored
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# load the TF-IDF vectorizer
loaded_tfidf = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

# supported models to load
model_files = {
    'svc': 'svc_classifier_model.pkl',
    'knn': 'knn_classifier_model.pkl',
    'rf': 'random_forest_classifier_model.pkl',
    'nb': 'naive_bayes_classifier_model.pkl',
    'lr': 'logistic_regression_model.pkl',
}

# load all classifiers into a dictionary
models = {
    name: joblib.load(os.path.join(models_dir, filename))
    for name, filename in model_files.items()
}


def make_prediction(email: str, model_name: str) -> str:
    model = models.get(model_name.lower())
    if not model:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {list(models.keys())}")

    # vectorize the input email
    email_vector = loaded_tfidf.transform([email])

    # convert to dense array if required by the model (e.g., SVC)
    if model_name.lower() in ["svc"]:
        email_vector = email_vector.toarray()

    # predict
    prediction = model.predict(email_vector)[0]
    result = "spam" if prediction == 1 else "ham"

    return f"The email is predicted as {result} using model: {model_name.upper()}."

