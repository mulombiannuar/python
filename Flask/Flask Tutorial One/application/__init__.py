from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate 
from application.config import DevelopmentConfig, ProductionConfig
from dotenv import load_dotenv
import os

# Initialize the database object
db = SQLAlchemy()

# Initialize the Migrate object
migrate = Migrate()

# Load environment variables from .env file
load_dotenv()

def create_app():
    
    # Initialize the Flask application
    app = Flask(__name__)

    # Decide config based on environment
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)
    
    # Initialize the database with app
    db.init_app(app)
    migrate.init_app(app, db)

    # Import and register Blueprints for modular structure
    from application.views.home import home
    from application.views.auth import auth
    from application.views.user import user
    from application.views.post import post

    app.register_blueprint(home, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(user, url_prefix='/users')
    app.register_blueprint(post, url_prefix='/posts')
    
    from application.models.user import User

    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    # Setup LoginManager for user session management
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login_page'  # Redirect to login page if not authenticated
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        # Define how to load a user from the database
        return User.query.get(int(id))

    return app
