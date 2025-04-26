from flask import Blueprint, render_template
from flask_login import current_user
from application.models.post import Post

home = Blueprint('home', __name__)


@home.route('/', methods=['GET'])
def home_page():
    return render_template('home/home.html')