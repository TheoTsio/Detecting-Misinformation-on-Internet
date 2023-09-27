'''
Our vectors are 100-dimensional.
It’s a problem to visualize them unless we do something to reduce their dimensionality.

We will use the TSNE, 
a technique to reduce the dimensionality of the vectors and create two components, one for the X axis and one for the Y axis on a scatterplot.
'''
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

def reduce_dimensions(model):
    num_components = 2 # number of dimensions to keep after compression

    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    tsne = TSNE(n_components=num_components, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_embeddings(x_vals, y_vals, labels):
    import plotly.graph_objects as go
    N=100
    colors = np.random.rand(N)
    sz = np.random.rand(N) * 30
    fig = go.Figure()
    fig.update_layout(title="Word2Vec - Visualizzazione embedding con TSNE")
    fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
    ))
    fig.write_image("fig2.png")
    fig.show()
    return fig


model = Word2Vec.load("healthword2vec.model")

x_vals, y_vals, labels = reduce_dimensions(model)

plot = plot_embeddings(x_vals, y_vals, labels)