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
            'gender': request.form['gender'],
            'part_time_job': request.form['part_time_job'],
            'extracurricular_participation': request.form['extracurricular_participation'],
            'diet_quality': request.form['diet_quality'],
            'parental_education_level': request.form['parental_education_level'],
            'internet_quality': request.form['internet_quality'],
            'age': int(request.form['age']),
            'study_hours_per_day': float(request.form['study_hours_per_day']),
            'social_media_hours': float(request.form['social_media_hours']),
            'netflix_hours': float(request.form['netflix_hours']),
            'attendance_percentage': float(request.form['attendance_percentage']),
            'sleep_hours': float(request.form['sleep_hours']),
            'exercise_frequency': int(request.form['exercise_frequency']),
            'mental_health_rating': int(request.form['mental_health_rating']),
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

