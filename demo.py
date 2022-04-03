from runtime import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
import dash_bootstrap_components as dbc

import numpy as np
import pickle as pk
import zipfile


# load figures

def read_pkzip(f):
    with zipfile.ZipFile(f) as zf:
        data = pk.load(zf.open(f.strip('.zip').split('/')[-1]))
    return data

fig1 = read_pkzip('figs/fig1_tsne.pk.zip')
fig2_performance = read_pkzip('figs/fig2_performance.pk.zip')
fig2_svm, fig2_knn, fig2_xgb = [fig2_performance[i] for i in fig2_performance.keys()]
model_results = read_pkzip('data/model_results.pk.zip')


# init dash app

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# main html

main = [
html.Div([

    dcc.Markdown('''
    # MNIST Dash demo

    In this demo, we explore machine learning classification of MNIST. MNIST is a database of handwritten digits which have been size-normalized in a fixed-size image (28x28 pixels). This dataset is commonly used to demonstrate both statistical and machine learning concepts.
    '''),
    html.Div([], style={'width': '800px', 'height': '300px', 'margin': 'auto', 'background-image': 'url("https://miro.medium.com/max/1400/1*LyRlX__08q40UJohhJG9Ow.png")'}),

    dcc.Markdown('''
    Our goal is to implement classification of the digits on the MNIST dataset using non-deep leaning methods. To accomplish this, we first reduce the overall dimensionality of the 28x28 handwritten images by centering the data; this refers to reducing the overall number of input features (784 image pixels) to a representation of 56 (row + column) summed pixel intensity values. The reasoning behind this simplification is to reduce the complexity and greatly reduce the training time of the models. We apply three different models, support vector machine, k-nearest neighbors, and xgboost on the processed MNIST data.

    '''),

    dcc.Markdown('''
    ### Data visualization and model performance

    To visualize the data, we apply a t-SNE dimensionality reduction to project the 56 input features into three dimensions. By doing so, we can now understand the scope of how each handwritten digit relate to one another.

    To visualize the resulting classification performance, the heatmap shows the number of correctly- and mis-classified digits. Hover over a cell to show the total number of classifications for a specific digit.

    '''),

    # tsne
    dcc.Graph(figure=fig1, style={'width': '600px', 'height': '600px', 'display': 'inline-block'}),

    # performance matrix
    html.Div([
        dcc.Graph(id='graph'),
        dcc.Dropdown(id='input', options=[{'label': 'SVM', 'value': 'svm'}, {'label': 'KNN', 'value': 'knn'}, {'label': 'XGBoost', 'value': 'xgboost'}],
                     value='svm')
    ], style={'width': '450px', 'height': '600px', 'display': 'inline-block'}),

    # prediction input
    html.Div([
        dcc.Markdown('''
        ### Testing the models

        To test our models, please input a number into the canvas space. This input data is first converted to the correct feature size (28x28) and then a gaussian blur is applied to liken the image to those of the MNIST dataset. Although accuracy and precision is above 90% for all three models when applied to the MNIST test set, the output generated from the canvas may not reflect that. This is likely due to the canvas digit input’s dissimilarity from the strictly processed MNIST digits.

        ''', style={'display': 'inline-block'}),

        html.Div([
            DashCanvas(id='canvas', lineWidth=12, width=224, height=224, hide_buttons=['zoom', 'pan', 'line', 'redo', 'pencil', 'rectangle', 'select'], lineColor='black', goButtonTitle='Predict'),
            dcc.Markdown(id='predictions')],
            style={'height': '500px', 'width': '500px', 'margin': 'auto', 'margin-right': '200px'})
        ])

    # prediction output

], style={'width': '1200px', 'margin': 'auto', 'margin-top': '50px'}),


]

# callbacks for inputs

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='input', component_property='value')
)
def update_div(input_value):
    if input_value == 'svm':
        return fig2_svm
    elif input_value == 'knn':
        return fig2_knn
    elif input_value == 'xgboost':
        return fig2_xgb


@app.callback(
    Output('predictions', 'children'),
    Input('canvas', 'json_data'),
)
def update_data(string):
    try:
        mask = parse_jsonstring(string, shape=(224, 224))
    except:
        return "Out of Bounding Box, click clear button and try again"
    if mask.any():
        x = generate_input_from_drawing(mask)
        svm = str(model_results['svm']['model'].predict(x)[0])
        knn = str(model_results['knn']['model'].predict(x)[0])
        xgb = str(model_results['xgboost']['model'].predict(x)[0])
        predictions = '''
        * support vector machine: **{}**
        * k-nearest neighbor: **{}**
        * xgboost: **{}**
        '''.format(svm, knn, xgb)
        return predictions

# run server

app.layout = html.Div(main)
if __name__ == '__main__':
    app.run_server(debug=True)
