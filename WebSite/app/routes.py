from app import application
from flask import render_template, flash, redirect
from app.forms import SarcasmForm

@application.route('/')
@application.route('/index')
def index():
    return render_template('home.html')

@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/sarcasm', methods=['GET', 'POST'])
def detect_sarcasm():
    form = SarcasmForm()
    if form.validate_on_submit():
        flash('Detect sarcasm for: {}, more_details={}'.format(form.user_input.data, form.more_details.data))
        return redirect('/index')
    return render_template('sarcasm.html', form=form)
