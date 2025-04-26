from flask import Blueprint, render_template, request, flash, redirect, url_for
from application import db   #means from __init__.py import db
from flask_login import login_required, logout_user, current_user

user = Blueprint('user', __name__)
