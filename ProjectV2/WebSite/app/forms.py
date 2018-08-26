from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired

class SarcasmForm(FlaskForm):
    user_input = StringField("Enter your sentence here", validators=[DataRequired()])
    sample_size = SelectField(u'Select the size of sample dataset to use (a bigger sample SHOULD be more accurate)',
        choices=[('25kFiles', '50,000 samples'),('15kFiles',"30,000 samples"),
        ('10kFiles','20,000 samples'), ('5kFiles', '10,000 samples'),
        ('2.5kFiles', '5,000 samples'), ('1kFiles', '2,000 samples'),
        ('500Files', '1,000 samples'),('250Files','500 samples'),
        ('100Files','200 samples')])
    more_details = BooleanField('Tick for advanced information')
    submit = SubmitField('Show me the sarcasm')
