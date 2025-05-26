from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate 
from application.config import DevelopmentConfig, ProductionConfig
from dotenv import load_dotenv
from flask_mail import Mail
import os

# initialize the database object
db = SQLAlchemy()

# initialize the migrate object
migrate = Migrate()

# initialize Flask-Mail for email functionality
mail = Mail()


# load environment variables from .env file
load_dotenv()

def create_app():
    
    # initialize the flask application
    app = Flask(__name__)

    # decide config based on environment
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)
    
    # initialize the database with app
    db.init_app(app)
    
    # initialize migrate with app and db
    migrate.init_app(app, db)
    
    # initialize the mail with app
    mail.init_app(app) 
    
    # import and register blueprints for modular structure
    from application.views.home import home
   
    app.register_blueprint(home, url_prefix='/')
    
    # create database tables if they don't exist
    with app.app_context():
        db.create_all()

    return app
