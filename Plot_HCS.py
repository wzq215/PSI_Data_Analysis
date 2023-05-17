"""
@Filename: Plot_HCS.py
@Aim: Plot HCS for a specific Carrington Rotation.
@Author: Ziqi Wu
@Date of Last Change: 2022-02-18
"""

import numpy as np
import pyvista
import plotly.offline as py
import plotly.graph_objects as go
from ps_read_hdf_3d import ps_read_hdf_3d


def plot_HCS_SCIH(crid, rotation=0, plot_SC=True,plot_IH=False):
    """
    :param crid: An int or float. Carrington Rotation Number
    :param rotation: rotate the HCS around the z axis centered at (0,0,0), Anti-clockwise.
    :return: trace1: trace (go.Mesh3d) for HCS(corona). trace2: trace (go.Mesh3d) for HCS(helio)
    """
    # crid=2246
    param = 'br002'
    if plot_SC:
        data_sc = ps_read_hdf_3d(crid, 'corona', param, periodicDim=3)
        r_sc = np.array(data_sc['scales1'])  # 201 in Rs, distance from sun
        t_sc = np.array(data_sc['scales2'])  # 150 in rad, latitude
        p_sc = np.array(data_sc['scales3'])  # 256 in rad, Carrington longitude
        br_sc = np.array(data_sc['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
        br_sc = br_sc * 2.205e5  # nT

        tv_sc, pv_sc, rv_sc = np.meshgrid(t_sc, p_sc, r_sc, indexing='xy')
    if plot_IH:
        data_ih = ps_read_hdf_3d(crid, 'helio', param, periodicDim=3)
        r_ih = np.array(data_ih['scales1'])  # 201 in Rs, distance from sun
        t_ih = np.array(data_ih['scales2'])  # 150 in rad, latitude
        p_ih = np.array(data_ih['scales3'])  # 256 in rad, Carrington longitude
        br_ih = np.array(data_ih['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
        br_ih = br_ih * 2.205e5  # nT

        tv_ih, pv_ih, rv_ih = np.meshgrid(t_ih, p_ih, r_ih, indexing='xy')

    def plot_HCS(rv, pv, tv, br, rotation):
        xv = rv * np.cos(pv) * np.sin(tv)
        yv = rv * np.sin(pv) * np.sin(tv)
        zv = rv * np.cos(tv)

        mesh = pyvista.StructuredGrid(xv, yv, zv)
        mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
        isos = mesh.contour(isosurfaces=1, rng=[0, 0])
        # isos.plot(opacity=0.7)
        if rotation != 0:
            axes = pyvista.Axes(show_actor=True, actor_scale=5.0, line_width=10)
            axes.origin = (0, 0, 0)
            isos = isos.rotate_z(rotation, point=axes.origin, inplace=False)
        vertices = isos.points
        triangles = isos.faces.reshape(-1, 4)
        trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                          opacity=0.5,  # colorscale='Viridis', cmax=15, cmin=-15,
                          i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                          # intensity=vertices[:, 2],
                          color='purple',
                          showscale=False,
                          )
        return trace

    return plot_HCS(rv_sc, pv_sc, tv_sc, br_sc, rotation), plot_HCS(rv_ih, pv_ih, tv_ih, br_ih, rotation)


def plot_HCS_rho_SCIH(crid, rotation=0):
    """
    :param crid: An int or float. Carrington Rotation Number
    :param rotation: rotate the HCS around the z axis centered at (0,0,0), Anti-clockwise.
    :return: trace1: trace (go.Mesh3d) for HCS(corona). trace2: trace (go.Mesh3d) for HCS(helio)
    """
    def plot_HCS_rho(crid, rotation=0, region='corona'):
        param = 'br002'
        data_br = ps_read_hdf_3d(crid, region, param, periodicDim=3)
        r_br = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
        t_br = np.array(data_br['scales2'])  # 150 in rad, latitude
        p_br = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
        br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
        br = br * 2.205e5  # nT

        param = 'rho002'
        data_rho = ps_read_hdf_3d(crid, region, param, periodicDim=3)
        r_rho = np.array(data_rho['scales1'])  # in Rs, distance from sun
        t_rho = np.array(data_rho['scales2'])  # in rad, latitude
        p_rho = np.array(data_rho['scales3'])  # in rad, Carrington longitude
        rho = np.array(data_rho['datas'])  # 1CU = 10^8 cm^-3
        rho = rho * 1e8  # cm^-3

        tv, pv, rv = np.meshgrid(t_br, p_br, r_br, indexing='xy')

        xv = rv * np.cos(pv) * np.sin(tv)
        yv = rv * np.sin(pv) * np.sin(tv)
        zv = rv * np.cos(tv)

        mesh = pyvista.StructuredGrid(xv, yv, zv)
        mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
        isos = mesh.contour(isosurfaces=1, rng=[0, 0])
        vertices = isos.points
        rho_points = vertices[:, 0] * 0
        for i in range(len(vertices)):
            point = np.array(vertices[i])
            r_p = np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
            # lon_carrington = np.arcsin(xyz_carrington[1]/np.sqrt(xyz_carrington[0]**2+xyz_carrington[1]**2))
            # if lon_carrington < 0:
            #     lon_carrington += 2*np.pi
            p_p = np.arcsin(point[1]/np.sqrt(point[0]**2+point[1]**2))
            if point[0] < 0:
                p_p = np.pi - p_p
            if p_p < 0:
                p_p +=2*np.pi
            # if point[1] > 0:
            #     p_p = np.arccos(point[0] / r_p)
            # elif point[1] <= 0:
            #     p_p = 2 * np.pi - np.arccos(point[0] / r_p)

            # t_p = np.pi - np.arccos(point[2] / r_p)
            t_p = np.arccos(point[2]/r_p)

            r_ind = np.argmin(abs(r_p - r_br))
            p_ind = np.argmin(abs(p_p - p_br))
            t_ind = np.argmin(abs(t_p - t_br))

            rho_points[i] = rho[p_ind, t_ind, r_ind] * (r_p) ** 2

        intensity = np.log10(np.array(rho_points)).reshape(-1, 1)

        if rotation != 0:
            axes = pyvista.Axes(show_actor=True, actor_scale=5.0, line_width=10)
            axes.origin = (0, 0, 0)
            isos = isos.rotate_z(rotation, point=axes.origin, inplace=False)
        vertices = isos.points
        triangles = isos.faces.reshape(-1, 4)



        trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                          opacity=0.5, colorscale='dense_r',
                          i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                          intensity=intensity,
                          cmax=7, cmin=5,
                          # color='purple',
                          # colorscale='Viridis',
                          # autocolorscale=True,
                          showscale=True,
                          colorbar=dict(
                              title=dict(text="log[ρr^2 (cm^-3 Rs)]",
                                         side='right',
                                         font=dict(size=25)),
                              tickmode="array",
                              # tickvals=[5,6],
                              # ticktext=["$10^5$", "$10^6$"],
                              ticks="outside",
                              thickness=50,
                              x=0.9,xanchor='left',
                              y=0.5,yanchor='middle',
                              ticklabelposition='outside',
                              tickfont=dict(size=25)
                          ),
                          # colorbar_title_font_size=20,
                          # colorbar_tickfont_size=20,
                          )
        return trace


    return plot_HCS_rho(crid, rotation=rotation, region='corona'), plot_HCS_rho(crid, rotation=rotation, region='helio')


if __name__ == '__main__':
    crid = 2148
    # plot = go.Figure()
    trace1, trace2 = plot_HCS_SCIH(crid)
    # plot.add_trace(trace1)
    # plot.add_trace(trace2)
    import plotly.io as pi

    plot = go.Figure(data=[trace1, trace2])
    plot.update_layout(
        title=dict(
            text="HCS (SC+IH)",
            y=0.9, x=0.5,
            xanchor='center',
            yanchor='top'),
        scene=dict(
            xaxis_title='X (Rs)',
            yaxis_title='Y (Rs)',
            zaxis_title='Z (Rs)', ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1, ),
    )
    # pi.to_html(plot,include_plotlyjs=False)
    py.plot(plot, filename='Figures/cr' + str(crid) + 'HCS.html')
    # plot.write_html('Figures/cr'+str(crid)+'HCS.html',include_plotlyjs=False)
