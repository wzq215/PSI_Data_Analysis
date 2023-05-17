"""
@Filename: Plot_Earth.py
@Aim: plot an animation including HCS(SC&IH), three Observers and correspondent field lines and the Earth in a Carrington Coordinate
@Author: Ziqi Wu
@Date of Last Change: 2022-02-18
"""
import os

import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd
from matplotlib import pyplot as plt

from img2zvals import image2zvals
# import pyvista
# from ps_read_hdf_3d import get_path



def topo_spheres(r, pos, opacity=1, planet='earth'):
    """
    :param r: A float. Radius of the sphere.
    :param pos: A vector [x,y,z]. Position of the center of the sphere
    :param opacity: A float between 0 and 1. Opacity.
    :param planet: A string. 'earth'/'venus'/'mercury'.
    :return: A trace (go.Surface()) for plotly.
    """

    theta = np.linspace(0, 2 * np.pi, 360)
    phi = np.linspace(0, np.pi, 180)
    tt, pp = np.meshgrid(theta, phi)

    x0 = pos[0] + r * np.cos(tt) * np.sin(pp)
    y0 = pos[1] + r * np.sin(tt) * np.sin(pp)
    z0 = pos[2] + r * np.cos(pp)

    # path = os.path.join(get_path(), 'Data')
    df = pd.read_csv('data/topo.CSV')
    lon = df.columns.values
    lon = np.array(float(s) for s in lon[1:])
    lat = np.array(df.iloc[:, 0].values)
    topo = np.array(df.iloc[:, 1:].values)

    if planet == 'earth':
        filename = 'land_shallow_topo_2048.jpeg' # https://raw.githubusercontent.com/empet/Datasets/master/Images/flowers.png
        img1 = plt.imread(filename)
        z_data, pl_colorscale = image2zvals(img1,n_colors=16,n_training_pixels=180)
        trace = go.Surface(x=x0,y=y0,z=z0,surfacecolor=z_data,colorscale=pl_colorscale,opacity=opacity,showscale=False)
        # trace = go.Surface(x=x0, y=y0, z=z0, surfacecolor=topo, colorscale='YlGnBu', opacity=opacity)
    elif planet == 'venus':
        filename = '46k-venus-color-map-3d-model.jpeg' # https://raw.githubusercontent.com/empet/Datasets/master/Images/flowers.png
        img1 = plt.imread(filename)
        z_data, pl_colorscale = image2zvals(img1,n_colors=16,n_training_pixels=180)
        trace = go.Surface(x=x0,y=y0,z=z0,surfacecolor=z_data,colorscale=pl_colorscale,opacity=opacity,showscale=False)
        # trace = go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2], intensity=z_data,intensitymode='cell', i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],colorscale=pl_colorscale, opacity=opacity)
    elif planet == 'mercury':
        filename = 'mercury.png' # https://raw.githubusercontent.com/empet/Datasets/master/Images/flowers.png
        img1 = plt.imread(filename)
        z_data, pl_colorscale = image2zvals(img1,n_colors=8,n_training_pixels=180)
        trace = go.Surface(x=x0,y=y0,z=z0,surfacecolor=z_data,colorscale=pl_colorscale,opacity=opacity,showscale=False)


    return trace


if __name__ == '__main__':
    plot = go.Figure()
    plot.add_trace(topo_spheres(1, [0, 0, 0], opacity=1,planet='mercury'))
    py.plot(plot)
