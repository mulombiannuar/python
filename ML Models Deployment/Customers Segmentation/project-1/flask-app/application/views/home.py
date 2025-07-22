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
            'annual_income_k': form.annual_income_k.data,
            'spending_score': form.spending_score.data,
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

