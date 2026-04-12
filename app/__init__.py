from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app():
    """Create and configure the Flask application"""
    # Get the root directory
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    app = Flask(__name__, 
                template_folder=os.path.join(root_dir, 'templates'),
                static_folder=os.path.join(root_dir, 'static'))
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///oil_optimization.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    instance_dir = os.path.join(root_dir, 'instance')
    os.makedirs(instance_dir, exist_ok=True)
    train_sess = os.path.join(instance_dir, 'train_sessions')
    os.makedirs(train_sess, exist_ok=True)
    app.config['TRAIN_SESSION_DIR'] = train_sess
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB uploads
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    from app.routes import bp as main_bp
    from app.train_routes import train_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(train_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
        # Initialize ML models on startup
        from app.utils import initialize_models
        initialize_models()
    
    return app
