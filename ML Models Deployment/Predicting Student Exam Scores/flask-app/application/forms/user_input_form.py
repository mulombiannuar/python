from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, FloatField
from wtforms.validators import DataRequired, NumberRange

class UserInputForm(FlaskForm):
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    
    part_time_job = SelectField('Part-Time Job', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    
    extracurricular_participation = SelectField('Extracurricular Participation', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    
    diet_quality = SelectField('Diet Quality', choices=[('Poor', 'Poor'), ('Fair', 'Fair'), ('Good', 'Good')], validators=[DataRequired()])
    
    parental_education_level = SelectField('Parental Education Level', choices=[
        ('High School', 'High School'), 
        ('Bachelor', 'Bachelor'), 
        ('Master', 'Master')
    ], validators=[DataRequired()])
    
    internet_quality = SelectField('Internet Quality', choices=[('Poor', 'Poor'), ('Average', 'Average'), ('Good', 'Good')], validators=[DataRequired()])
    
    age = IntegerField('Age', validators=[
        DataRequired(), 
        NumberRange(min=10, max=100, message="Enter a valid age.")
    ])
    
    study_hours_per_day = FloatField('Study Hours Per Day', validators=[
        DataRequired(), 
        NumberRange(min=0, max=24)
    ])
    
    social_media_hours = FloatField('Social Media Hours per Day', validators=[
        DataRequired(), 
        NumberRange(min=0, max=24)
    ])
    
    netflix_hours = FloatField('Netflix Hours per Day', validators=[
        DataRequired(), 
        NumberRange(min=0, max=24)
    ])
    
    attendance_percentage = FloatField('Attendance Percentage', validators=[
        DataRequired(), 
        NumberRange(min=0, max=100)
    ])
    
    sleep_hours = FloatField('Sleep Hours per Day', validators=[
        DataRequired(), 
        NumberRange(min=0, max=24)
    ])
    
    exercise_frequency = IntegerField('Exercise Frequency (days/week)', validators=[
        DataRequired(), 
        NumberRange(min=0, max=7)
    ])
    
    mental_health_rating = IntegerField('Mental Health Rating (1-5)', validators=[
        DataRequired(), 
        NumberRange(min=1, max=5)
    ])
