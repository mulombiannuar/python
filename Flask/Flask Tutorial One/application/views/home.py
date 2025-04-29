from flask import Blueprint, render_template
from flask_login import login_required, current_user
from application.forms.user_form import RegistrationForm

home = Blueprint('home', __name__)


@home.route('/', methods=['GET'])
def home_page():
    form = RegistrationForm()
    return render_template('auth/login.html', form=form)


@home.route('/dashboard', methods=['GET'])
@login_required
def dashboard_page():
    return render_template('home/dashboard.html')