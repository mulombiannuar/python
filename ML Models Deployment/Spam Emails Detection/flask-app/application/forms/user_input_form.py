from flask_wtf import FlaskForm
from wtforms import TextAreaField, SelectField
from wtforms.validators import DataRequired, Length

class UserInputForm(FlaskForm):
    email_text = TextAreaField('Email Text', validators=[
        DataRequired(message="Email content is required."),
        Length(min=10, message="Email content is too short.")
    ])

    model_choice = SelectField('Model', choices=[
        ('svc', 'Support Vector Classifier (SVC)'),
        ('knn', 'K-Nearest Neighbors'),
        ('rf', 'Random Forest'),
        ('nb', 'Naive Bayes'),
        ('lr', 'Logistic Regression')
    ], validators=[DataRequired(message="Please select a model.")])
