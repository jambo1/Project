from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class SarcasmForm(FlaskForm):
    user_input = StringField('Input to get sarcasm score', validators=[DataRequired()])
    more_details = BooleanField('Tick for advanced information')
    submit = SubmitField('Show me the sarcasm')