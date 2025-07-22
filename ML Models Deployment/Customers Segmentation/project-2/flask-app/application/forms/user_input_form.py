from flask_wtf import FlaskForm
from wtforms import IntegerField
from wtforms.validators import DataRequired, NumberRange

class UserInputForm(FlaskForm):
    income = IntegerField('Annual Income ($)', validators=[
        DataRequired(message="Income is required."),
        NumberRange(min=0, max=1000000, message="Enter a valid income.")
    ])
    
    age = IntegerField('Age', validators=[
        DataRequired(message="Age is required."),
        NumberRange(min=1, max=120, message="Enter a valid age.")
    ])

    total_spending = IntegerField('Total Spending ($)', validators=[
        DataRequired(message="Total Spending is required."),
        NumberRange(min=0, message="Spending must be non-negative.")
    ])

    recency = IntegerField('Recency (days)', validators=[
        DataRequired(message="Recency is required."),
        NumberRange(min=0, message="Recency must be non-negative.")
    ])

    num_web_purchases = IntegerField('Number of Web Purchases', validators=[
        DataRequired(message="This field is required."),
        NumberRange(min=0, message="Must be non-negative.")
    ])

    num_store_purchases = IntegerField('Number of Store Purchases', validators=[
        DataRequired(message="This field is required."),
        NumberRange(min=0, message="Must be non-negative.")
    ])

    accepted_any = IntegerField('Accepted Any Offer (0 or 1)', validators=[
        DataRequired(message="This field is required."),
        NumberRange(min=0, max=1, message="Must be 0 or 1.")
    ])

    num_web_visits_month = IntegerField('Web Visits per Month', validators=[
        DataRequired(message="This field is required."),
        NumberRange(min=0, max=100, message="Enter a realistic number.")
    ])
