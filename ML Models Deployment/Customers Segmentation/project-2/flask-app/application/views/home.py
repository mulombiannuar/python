from flask import Blueprint, render_template, request, jsonify
from application.forms.user_input_form import UserInputForm 
from application.services.prediction import make_prediction

# create home blueprint
home = Blueprint('home', __name__)

# define the home route
@home.route('/', methods=['GET'])
def home_page():
    form = UserInputForm()
    return render_template('home/home.html', form=form)


# define the predict route
@home.route('/predict', methods=['POST', 'GET'])
def predict_outcome():
    form = UserInputForm()

    if request.method == 'POST' and form.validate_on_submit(): 
        data = {
            'Income': [form.income.data],
            'Age': [form.age.data],
            'Total_Spending': [form.total_spending.data],
            'Recency': [form.recency.data],
            'NumWebPurchases': [form.num_web_purchases.data],
            'NumStorePurchases': [form.num_store_purchases.data],
            'AcceptedAny': [form.accepted_any.data],
            'NumWebVisitsMonth': [form.num_web_visits_month.data]
        }

        # prediction
        prediction = make_prediction(data)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    # if validation fails or method is not POST
    return jsonify({
        "success": False,
        "message": "Invalid input. Please check your form."
    }), 400


