from app import application
from flask import render_template
from app.forms import SarcasmForm

@application.route('/')
@application.route('/index')
def index():
    return render_template('home_page.html')

@application.route('/sarcasm')
def login():
    form = SarcasmForm()
    return render_template('sarcasm.html', form=form)
