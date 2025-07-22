from flask import Flask
from application.config import DevelopmentConfig, ProductionConfig
from dotenv import load_dotenv
import os

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

    # import and register blueprints for modular structure
    from application.views.home import home
   
    app.register_blueprint(home, url_prefix='/')
    
    return app
