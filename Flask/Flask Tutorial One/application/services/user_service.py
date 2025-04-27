from application import db
from application.models.user import User
from werkzeug.security import generate_password_hash
from sqlalchemy.exc import SQLAlchemyError

class UserService:
    
    @staticmethod
    def create_user(data):
        try:
            hashed_password = generate_password_hash(data['password'])
            user = User(
                email=data['email'],
                password=hashed_password,
                first_name=data['first_name'],
                last_name=data['last_name'],
                gender=data['gender'],
                address=data.get('address', ''),
                zip_code=data.get('zip_code', '')
            )
            db.session.add(user)
            db.session.commit()
            return user
        except SQLAlchemyError as e:
            db.session.rollback()
            print(f"Error creating user: {str(e)}")
            return None

   
    @staticmethod
    def get_user_by_id(user_id):
        try:
            return User.query.get(user_id)
        except SQLAlchemyError as e:
            print(f"Error fetching user: {str(e)}")
            return None
        
        
    @staticmethod
    def get_user_by_email(email):
        try:
            return User.query.filter_by(email=email).first()  
        except SQLAlchemyError as e:
            print(f"Error fetching user: {str(e)}")
            return None

        
    @staticmethod
    def update_user(user_id, data):
        try:
            user = User.query.get(user_id)
            if not user:
                return None

            user.first_name = data.get('first_name', user.first_name)
            user.last_name = data.get('last_name', user.last_name)
            user.email = data.get('email', user.email)
            user.gender = data.get('gender', user.gender)
            user.address = data.get('address', user.address)
            user.zip_code = data.get('zip_code', user.zip_code)

            if 'password' in data and data['password']:
                user.password = generate_password_hash(data['password'])

            db.session.commit()
            return user
        except SQLAlchemyError as e:
            db.session.rollback()
            print(f"Error updating user: {str(e)}")
            return None

    @staticmethod
    def delete_user(user_id):
        try:
            user = User.query.get(user_id)
            if not user:
                return False
            db.session.delete(user)
            db.session.commit()
            return True
        except SQLAlchemyError as e:
            db.session.rollback()
            print(f"Error deleting user: {str(e)}")
            return False
