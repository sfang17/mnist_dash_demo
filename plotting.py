import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pickle as pk
import numpy as np

# useful methods





# generating plots needs data from get_mnist_dataset()
def generate_3d_tsne(tsne_projections, y):
    # generates figure
    fig = px.scatter_3d(
        tsne_projections, x=0, y=1, z=2,
        color=y.astype(str), labels={'color': 'Digit'}
    )
    # updates point size
    fig.update_traces(marker={'size': 3})
    # removes axis labels
    for i in ['xaxis', 'yaxis', 'zaxis']:
        fig['layout']['scene'][i]={'showticklabels': False}
    # corrects figure size
    fig.update_layout(template='plotly_white')
    return fig





# generates the missclassification matrix
def generate_heatmap(contact_mat):
    # plots the matrix
    fig = go.Figure(data=go.Heatmap(x=[str(i) for i in range(10)],
                                        y=[str(i) for i in range(10)],
                                        z=np.log1p(contact_mat),
                                        text=contact_mat,
                                        texttemplate="%{text}",
                                        textfont={"size": 10}))
    # removes legend
    fig.data[0]['showscale']=False
    # fixes hover
    fig.data[0]['hovertemplate']='Predicted: %{y} - True: %{x} <br> Total: %{text}<extra></extra>'
    return fig






# runit
if __name__ == '__main__':
    print('generating tsne figs...')
    # generates the tsne figs
    with open('data/tsne_projections_labels.pk', 'rb') as _:
        tsne_projections_labels = pk.load(_)
    tsne_projections, y = tsne_projections_labels
    fig1_tsne = generate_3d_tsne(tsne_projections, y)
    with open('figs/fig1_tsne.pk', 'wb') as _:
        pk.dump(fig1_tsne, file = _)
    
    print('generating performance figs...')
    # generates the heatmap
    with open('data/model_performance.pk', 'rb') as _:
        model_performance = pk.load(_)
    fig2_performance = {}
    for i in model_performance.keys():
        fig2_performance[i] = generate_heatmap(model_performance[i])
    with open('figs/fig2_performance.pk', 'wb') as _:
        pk.dump(fig2_performance, file = _)
    












#
