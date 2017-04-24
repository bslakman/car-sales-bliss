import requests
from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re
import os.path
from ediblepickle import checkpoint

from bokeh.charts import Bar
from bokeh.layouts import row
from bokeh.models import LinearAxis, HoverTool
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure, show

from sklearn.externals import joblib

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

region_dict = {'cap': 'Cape Cod/Islands',
'har': 'Hartford',
'new': 'Eastern CT',
'pro': 'Rhode Island',
'sou': 'South Coast',
'wor': 'Worcester/Central Mass',
'bmw': 'Boston Metro West',
'gbs': 'Boston/Cambridge/Brookline',
'nos': 'North Shore',
'nwb': 'Northwest/Merrimack',
'sob': 'South Shore'}

def fetch(query = None, auto_make_model = None, min_auto_year = None, max_auto_year = None, s=0):
    search_params = {key: val for key, val in locals().items() if val is not None}
    if not search_params:
        raise ValueError("No valid keywords")

    base = "http://boston.craigslist.org/search/cto"
    resp = requests.get(base, params=search_params, timeout=3)
    resp.raise_for_status()
    return resp.content, resp.encoding

def parse(html, encoding='utf-8'):
    parsed = BeautifulSoup(html, 'lxml', from_encoding=encoding)
    return parsed

def extract_listings(parsed):
    listings = parsed.find_all('p', class_='result-info')
    extracted = []
    for listing in listings:
        title = listing.find('a', class_='result-title hdrlnk')
        price = listing.find('span', class_='result-price')
        try:
            price_string = price.string.strip()
        except AttributeError:
            price_string = ''
        location = listing.find('span', class_='result-hood')
        try:
            loc_string = location.string.strip()[1:-1].split()[0]
        except AttributeError:
            loc_string = ''
        this_listing = {
            'link': title.attrs['href'],
            'description': title.string.strip(),
            'price': price_string,
            'location': loc_string
        }
        extracted.append(this_listing)
    return extracted

def get_mileage(description):
    description = description.lower().split('k miles')
    if len(description) == 1:
        description = description[0].split('000 miles')
        if len(description) == 1:
            try:
                description = re.search('(\d{1,3})k', description[0]).groups()
            except:
                return np.nan
    mileage = re.sub('[^0-9]', '', description[0].split()[-1])
    try:
        mileage = int(mileage) * 1000
        return mileage
    except:
        return np.nan

def get_year(description):
    description = re.split('(20[0-9][0-9])', description)
    if len(description) == 1:
        description = re.split('([0-1][0-9])', description[0])
    try:
        return int(description[1]) if len(description[1]) == 4 else int('20' + description[1])
    except:
        return np.nan

def get_standard_location(location):
    """
    Use first 5 characters of location in order to group. Gets rid of much of the weird stuff
    """
    if len(location) < 5:
        return location.lower()
    else:
        return location[:5].lower()

def get_price(price):
    try:
        return int(price[1:])
    except:
        return np.nan

def draw_regional_fig(make, model, year):
    listings = []
    make_model = "{0} {1}".format(make,model)
    min_auto_year = int(year) - 2
    max_auto_year = int(year) + 2
    if max_auto_year > 2016:
        max_auto_year = 2016
    for i in range(0, 500, 100):
        car_results = fetch(auto_make_model=make_model, min_auto_year=min_auto_year, max_auto_year=max_auto_year, s=i)
        doc = parse(car_results[0])
        listings.extend(extract_listings(doc))

    df = pd.DataFrame(data=listings)
    if len(df) == 0: return "No results found, check your spelling"
    df['mileage'] = df.apply(lambda row: get_mileage(row['description']), axis=1)
    df['price'] = df.apply(lambda row: get_price(row['price']), axis=1)
    df['region'] = df['link'].str[1:5]
    df['year'] = df.apply(lambda row: get_year(row['description']), axis=1)

    regions = df.groupby('region').mean()
    regions = regions.append(pd.Series(data={'year': np.mean(df['year']), 'price': np.mean(df['price']), 'mileage': np.mean(df['mileage'])}, name='AVERAGE'))

    sns.set_style('ticks')
    my_title = 'Average Price and Mileage of Used {0} {1}, \n{2}-{3}, by region, n={4}'.format(make, model, min_auto_year, max_auto_year, len(df))
    ax = regions['price'].plot.bar(position=0, width=0.3, alpha=0.5, legend=True, title=my_title, figsize=(5,3))
    ax.set_ylabel('Price($)')
    ax = regions['mileage'].plot.bar(secondary_y=True, color='green', position=1, width=0.3, alpha=0.5, legend=True)
    ax.set_ylabel('Mileage')
    sns.despine(top=True, right=False)
    fig=ax.get_figure()
    fig.set_tight_layout(True)
    return fig

def draw_regional_fig_bokeh(make, model, year):
    try:
        min_auto_year = int(year) - 2
        max_auto_year = int(year) + 2
    except:
        min_auto_year = 2012
        max_auto_year = 2016

    listings = get_listings(make, model, year)

    df = pd.DataFrame(data=listings)
    if len(df) == 0: return "No results found, check your spelling"
    df['mileage'] = df.apply(lambda row: get_mileage(row['description']), axis=1)
    df['price'] = df.apply(lambda row: get_price(row['price']), axis=1)
    df['region'] = df['link'].str[1:5]
    df['year'] = df.apply(lambda row: get_year(row['description']), axis=1)
    df['full_region'] = df.apply(lambda row: region_dict[row['region'].strip('/')], axis=1)
    mileage_count = len(df[df['mileage'].notnull()])

    title1 = 'Average Price of Used {0} {1}, {2}-{3}, by region, n={4}'.format(make, model, min_auto_year, max_auto_year, len(df))
    title2 = 'Average Mileage of Used {0} {1}, {2}-{3}, by region, n={4}'.format(make, model, min_auto_year, max_auto_year, mileage_count)
    bar1 = Bar(df, label='full_region', values='price', agg='mean',
        title=title1, legend=None, xlabel="Region", ylabel="Price", y_range=(0, 20000))
    bar2 = Bar(df, label='full_region', values='mileage', agg='mean',
        title=title2, legend=None, xlabel="Region", ylabel="Mileage", y_range=(0, 150000), color='dodgerblue')
    return row(bar1, bar2)

@checkpoint(key=lambda args, kwargs: "-".join(args) + '.p', work_dir=cache_dir)
def get_listings(make,model,year):
    listings = []
    make_model = "{0} {1}".format(make,model)
    try:
        min_auto_year = int(year) - 2
        max_auto_year = int(year) + 2
    except:
        min_auto_year = 2012
        max_auto_year = 2016
    if max_auto_year > 2016:
        max_auto_year = 2016
    for i in range(0, 500, 100):
        car_results = fetch(auto_make_model=make_model, min_auto_year=min_auto_year, max_auto_year=max_auto_year, s=i)
        doc = parse(car_results[0])
        listings.extend(extract_listings(doc))
    return listings

def predict(make='', model='', year='', mileage='', title_status='', color='', body_type=''):
    X_test = [{'make': make.lower(),
             'model': model.lower(),
             'year': int(year) if year else 0,
             'odometer': int(mileage) if mileage else 0,
             'title status': title_status,
             'paint color': color,
             'type': body_type,
             }]
    v = joblib.load('Cars/dv.pkl')
    n = joblib.load('Cars/mas.pkl')
    X_trans = v.transform(X_test)
    X_norm = n.transform(X_trans)
    kn = joblib.load('Cars/kneighbors_kn.pkl')
    all_filtered = joblib.load('Cars/all_filtered.pkl')
    price = kn.predict(X_norm[0])
    neighbors_info = []
    for j in kn.kneighbors(X_norm[0])[1][0][:10]:
        neighbors_info.append((all_filtered.iloc[j]['description'],all_filtered.iloc[j]['link'],all_filtered.iloc[j]['price']))
    return price[0], neighbors_info
