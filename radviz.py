# Radviz code
# Mimics logic but not actually calling matplotlib's radviz

import pandas as pd
import plotly.express as px
import numpy as np

# Not yet working, if we end up going for radviz will try and get this to work for interaction
# For demo will use a simpler function below
def get_radviz_coords(frame, class_column):
    # mimic the internal logic to get x,y
    classes = frame[class_column].unique()
    n = len(frame.columns) - 1
    s = pd.Series(index=frame.columns.drop(class_column), dtype=float)
    theta = 2 * np.pi * np.arange(n) / n
    anchors = np.c_[np.cos(theta), np.sin(theta)]
    for i, c in enumerate(frame.columns.drop(class_column)):
        s[c] = i
    X = frame.drop(columns=[class_column])
    X = (X - X.min()) / (X.max() - X.min())
    # Compute coordinates
    x = X.dot(anchors[:, 0]) / X.sum(axis=1)
    y = X.dot(anchors[:, 1]) / X.sum(axis=1)

    return pd.DataFrame({"x": x, "y": y, class_column: frame[class_column]})

import plotly.tools as tls
from pandas.plotting import radviz
import matplotlib.pyplot as plt


def radviz_from_mpl(data, min_year, max_year):
    # Create matplotlib Radviz
    fig_mpl, ax = plt.subplots()
    radviz(data, "state", ax=ax)

    # Convert to Plotly
    fig_plotly = tls.mpl_to_plotly(fig_mpl)
    fig_plotly.update_layout(
        title={
            'text': f'Overview {min_year}-{max_year}',
            'font': {'size': 16, 'family': 'Arial, sans-serif', 'weight': 'bold'},
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    return fig_plotly