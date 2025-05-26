from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField
from wtforms.validators import DataRequired, NumberRange
from flask_wtf.file import FileAllowed

class UserInputForm(FlaskForm):
    pregnant = IntegerField('Number of Pregnancies', validators=[
        DataRequired(message="Number of Pregnancies is required."),
        NumberRange(min=0, max=20, message="Must be between 0 and 20.")
    ])
    glucose = FloatField('Glucose Level', validators=[
        DataRequired(message="Glucose Level is required."),
        NumberRange(min=0, max=300, message="Must be between 0 and 300.")
    ])
    blood_pressure = FloatField('Blood Pressure', validators=[
        DataRequired(message="Blood Pressure is required."),
        NumberRange(min=0, max=200, message="Must be between 0 and 200.")
    ])
    skin_thickness = FloatField('Skin Thickness', validators=[
        DataRequired(message="Skin Thickness is required."),
        NumberRange(min=0, max=100, message="Must be between 0 and 100.")
    ])
    insulin = FloatField('Insulin Level', validators=[
        DataRequired(message="Insulin Level is required."),
        NumberRange(min=0, max=1000, message="Must be between 0 and 1000.")
    ])
    bmi = FloatField('BMI', validators=[
        DataRequired(message="BMI is required."),
        NumberRange(min=0, max=100, message="Must be between 0 and 100.")
    ])
    diabetes_pedigree = FloatField('Diabetes Pedigree', validators=[
        DataRequired(message="Diabetes Pedigree is required."),
        NumberRange(min=0.0, max=2.5, message="Must be between 0.0 and 2.5.")
    ])
    age = IntegerField('Age', validators=[
        DataRequired(message="Age is required."),
        NumberRange(min=1, max=120, message="Must be between 1 and 120.")
    ])