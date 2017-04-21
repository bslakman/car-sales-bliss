from craigslist import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from bokeh.embed import components

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
        app.vars['status'] = request.form['Title Status']
        app.vars['color'] = request.form['Color']
        app.vars['type'] = request.form['Body Type']

        return redirect('/result')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method=='GET':
        graph = draw_regional_fig_bokeh(app.vars['make'], app.vars['model'], app.vars['year'])
        script, div = components(graph)

        price, similar = predict(make=app.vars['make'], model=app.vars['model'],
            year=app.vars['year'], mileage=app.vars['mileage'], title_status=app.vars['status'],
            color=app.vars['color'], body_type=app.vars['type'])

        price = '${}'.format(int(round(price)))
        similar_html = ''
        for car in similar:
            desc = car[0]
            link = car[1]
            car_price = car[2]
            similar_html += '<a href="https://boston.craigslist.org/{0}">{1}</a> ${2}<br>'.format(
                link, desc, int(car_price))

        return render_template('car_result.html', script=script, div=div, similar_html=similar_html, price=price)
    else:
        return redirect('/index')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 33507))
    app.run(port=port)
