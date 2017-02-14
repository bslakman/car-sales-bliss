from craigslist import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO
from flask import Flask, render_template, request, redirect, send_file
app = Flask(__name__)

app.vars={}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('car_info.html')
    else:
        app.vars['make'] = request.form['Make']
        app.vars['model'] = request.form['Model']
        app.vars['year'] = request.form['Year']
        app.vars['mileage'] = request.form['Mileage']

        return redirect('/result')

@app.route('/result')
def result():
    return render_template('car_result.html')

@app.route('/fig/')
def fig():
    fig = draw_regional_fig(app.vars['make'], app.vars['model'], app.vars['year'])
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run()
