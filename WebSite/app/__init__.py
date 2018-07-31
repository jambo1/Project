from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap

application = Flask(__name__)
application.config.from_object(Config)
bootstrap = Bootstrap(application)

from app import routes