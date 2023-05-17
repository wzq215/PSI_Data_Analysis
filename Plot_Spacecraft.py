"""
@Filename: Plot_Spacecraft.py
@Aim: load an stl file and visualize it with appropriate scaling and rotation.
@Author: Ziqi Wu
@Date of Last Change: 2022-02-18
"""
import os

import pyvista as pv
import numpy as np
from plotly import graph_objects as go
import plotly.offline as py
import spiceypy as spice
from datetime import datetime
import furnsh_kernels
# from ps_read_hdf_3d import get_path


def add_texture(pos, rot_theta, scale=10):
    """
    :param pos: A vector [x,y,z]. Center position of the spacecraft.
    :param rot_theta: A float (deg). Rotation around the z axis centered at pos.
    :param scale: A float, 10 by default. Scaling of the spacecraft.
    :return: A trace (go.Mesh3d()) for plotly.
    """
    # path = os.path.join(get_path(), 'Data')
    # mesh = pv.read(path+'/Stereo.stl')

    mesh = pv.read('data/ParkerSolarProbe.stl')
    print(mesh)
    # mesh.plot()
    # scale = 10
    mesh.points = scale * mesh.points / np.max(mesh.points)
    theta_x = 270 # Red
    theta_z = 90+rot_theta # Green
    theta_y = 0 # Yellow
    axes = pv.Axes(show_actor=True, actor_scale=5.0, line_width=10)
    axes.origin = (0, 0, 0)
    rot = mesh.rotate_x(theta_x, point=axes.origin, inplace=False)
    rot = rot.rotate_z(theta_z, point=axes.origin, inplace=False)
    rot = rot.rotate_y(theta_y, point=axes.origin, inplace=False)
    # rot = mesh
    # Visualize by Pyvista
    # p = pv.Plotter()
    # p.add_actor(axes.actor,pickable=True)
    # p.add_mesh(rot)
    # # p.add_mesh(mesh)
    # p.show()

    vertices = rot.points
    print(rot.points.shape)
    print(rot.faces.shape)
    triangles = rot.faces.reshape(-1, 4)
    trace = go.Mesh3d(x=vertices[:, 0] + pos[0], y=vertices[:, 1] + pos[1], z=vertices[:, 2] + pos[2],
                      opacity=1,
                      color='gold',
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      )
    return trace

def plot_span_a_ion(pos, rot_theta, dt,scale=10,):

    mesh = pv.read('/Users/ephe/Desktop/SolHelio-Viewer/ParkerSolarProbe.stl')
    mesh.points = scale * mesh.points / np.max(mesh.points)

    theta_x = 180 # Red
    theta_z = 90 # Green
    theta_y = 0 # Yellow
    axes = pv.Axes(show_actor=True, actor_scale=5.0, line_width=10)
    axes.origin = (0, 0, 0)

    # 旋转使得XYZ轴与SPP_SPACECRAFT一致
    rot = mesh.rotate_x(theta_x, point=axes.origin, inplace=False)
    rot = rot.rotate_z(theta_z, point=axes.origin, inplace=False)
    # rot = rot.rotate_y(theta_y, point=axes.origin, inplace=False)

    spana_ion_center = np.array([0.128,-0.0298,-0.293])*scale

    # # Visualize by Pyvista
    # p = pv.Plotter()
    # p.add_actor(axes.actor,pickable=True)
    # p.add_mesh(rot)
    # # p.add_mesh(mesh)
    # p.show()

    # 将pyvista中的mesh转换为可以用plotly可视化的形式
    vertices = rot.points
    triangles = rot.faces.reshape(-1, 4)

    plot = go.Figure()
    # 画PSP全体
    trace = go.Mesh3d(x=vertices[:, 0] + pos[0], y=vertices[:, 1] + pos[1], z=vertices[:, 2] + pos[2],
                      opacity=0.8,
                      color='silver',
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      lighting=dict(ambient=0.4,diffuse=1,roughness=1)
                      )
    plot.add_trace(trace)

    # 画SPAN-A FOV
    sweap_span_a_ion_parameter = spice.getfov(-96201, 26)
    print(sweap_span_a_ion_parameter)
    # edges = np.array([sweap_span_a_ion_parameter[4][0],
    #                   sweap_span_a_ion_parameter[4][12],
    #                   sweap_span_a_ion_parameter[4][13],
    #                   sweap_span_a_ion_parameter[4][25]])

    # 画所有的角
    edges = np.array(sweap_span_a_ion_parameter[4][:])

    et = spice.datetime2et(dt)
    M_arr_ION = spice.sxform('SPP_SWEAP_SPAN_A_ION','SPP_SPACECRAFT',et)[0:3,0:3]

    for i, edge in enumerate(edges):
        length_ray = 2
        tmp_ray = np.dot(M_arr_ION,edge)*length_ray
        R = np.linalg.norm(tmp_ray,2)
        # 画各FOV的四角
        plot.add_trace(go.Scatter3d(x=[spana_ion_center[0],spana_ion_center[0]+tmp_ray[0]],
                                    y=[spana_ion_center[1],spana_ion_center[1]+tmp_ray[1]],
                                    z=[spana_ion_center[2],spana_ion_center[2]+tmp_ray[2]],
                                    mode = 'lines',
                                    name = 'V'+str(i),
                                    legendgroup="group_ion",
                                    legendgrouptitle_text="SWEAP_SPAN_A_ION",
                                    showlegend=False,
                                    line=dict(color='blue',width=5)))

        # 连结不同的FOV
        if i != 12 and i != 25:
            next_ray = np.dot(M_arr_ION,edges[i+1])*length_ray
            # plot.add_trace(go.Scatter3d(x=[spana_ion_center[0]+tmp_ray[0],spana_ion_center[0]+next_ray[0]],
            #                             y=[spana_ion_center[1]+tmp_ray[1],spana_ion_center[1]+next_ray[1]],
            #                             z=[spana_ion_center[2]+tmp_ray[2],spana_ion_center[2]+next_ray[2]],
            #                             mode = 'lines',
            #                             showlegend=False,
            #                             line=dict(color='blue')))
            tmp_trace = plot_arc(spana_ion_center,R,spana_ion_center+tmp_ray,spana_ion_center+next_ray,longer_arc=False)
            tmp_trace.update(line=dict(color='blue'))
            plot.add_trace(tmp_trace)


        if i < 13:
            next_ray = np.dot(M_arr_ION,edges[25-i])*length_ray
            # plot.add_trace(go.Scatter3d(x=[spana_ion_center[0]+tmp_ray[0],spana_ion_center[0]+next_ray[0]],
            #                             y=[spana_ion_center[1]+tmp_ray[1],spana_ion_center[1]+next_ray[1]],
            #                             z=[spana_ion_center[2]+tmp_ray[2],spana_ion_center[2]+next_ray[2]],
            #                             mode = 'lines',
            #                             showlegend=False,
            #                             line=dict(color='blue')))
            tmp_trace = plot_arc(spana_ion_center,R,spana_ion_center+tmp_ray,spana_ion_center+next_ray,longer_arc=False)
            tmp_trace.update(line=dict(color='blue'))
            plot.add_trace(tmp_trace)


    # 标记SPAN-A中心
    plot.add_trace(go.Scatter3d(x=[spana_ion_center[0]],y=[spana_ion_center[1]],z=[spana_ion_center[2]],
                                mode='markers',
                                name='SWEAP_SPAN_A_ION',
                                marker=dict(size=5,
                                            color='blue',
                                            symbol='diamond')))
    plot.update_layout(scene_aspectmode='data',)
    py.plot(plot,filename='SPAN-A_Ion.html')
    # return trace

def plot_span_a_electron(pos, rot_theta, dt,scale=10):

    mesh = pv.read('/Users/ephe/Desktop/SolHelio-Viewer/ParkerSolarProbe.stl')
    mesh.points = scale * mesh.points / np.max(mesh.points)

    theta_x = 90 # Red
    theta_z = 90 # Green
    theta_y = 0 # Yellow
    axes = pv.Axes(show_actor=True, actor_scale=5.0, line_width=10)
    axes.origin = (0, 0, 0)

    # 旋转使XYZ轴与SPP_SPACECRAFT一致
    rot = mesh.rotate_x(theta_x, point=axes.origin, inplace=False)
    rot = rot.rotate_z(theta_z, point=axes.origin, inplace=False)
    rot = rot.rotate_y(theta_y, point=axes.origin, inplace=False)

    spana_electron_center = np.array([0.1040395,-0.0940903,-0.29254955])*scale

    # # Visualize by Pyvista
    # p = pv.Plotter()
    # p.add_actor(axes.actor,pickable=True)
    # p.add_mesh(rot)
    # # p.add_mesh(mesh)
    # p.show()

    # 将pyvista的mesh转换为plotly可视化的形式
    vertices = rot.points
    triangles = rot.faces.reshape(-1, 4)

    plot = go.Figure()

    # PSP主体
    trace = go.Mesh3d(x=vertices[:, 0] + pos[0], y=vertices[:, 1] + pos[1], z=vertices[:, 2] + pos[2],
                      opacity=0.9,
                      color='silver',
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      lighting=dict(ambient=0.4,diffuse=1,roughness=1)
                      )
    plot.add_trace(trace)

    sweap_span_a_electron_parameter = spice.getfov(-96202, 26)
    print(sweap_span_a_electron_parameter)
    # edges = np.array([sweap_span_a_electron_parameter[4][0],
    #                   sweap_span_a_electron_parameter[4][12],
    #                   sweap_span_a_electron_parameter[4][13],
    #                   sweap_span_a_electron_parameter[4][25]])
    edges = np.array(sweap_span_a_electron_parameter[4][:])

    et = spice.datetime2et(dt)
    M_arr_ION = spice.sxform('SPP_SWEAP_SPAN_A_ELECTRON','SPP_SPACECRAFT',et)[0:3,0:3]

    for i, edge in enumerate(edges):
        length_ray=2
        tmp_ray = np.dot(M_arr_ION,edge)*length_ray
        R = np.linalg.norm(tmp_ray,2)
        # 画各个FOV的四角
        plot.add_trace(go.Scatter3d(x=[spana_electron_center[0],spana_electron_center[0]+tmp_ray[0]],
                                    y=[spana_electron_center[1],spana_electron_center[1]+tmp_ray[1]],
                                    z=[spana_electron_center[2],spana_electron_center[2]+tmp_ray[2]],
                                    mode = 'lines',
                                    name = 'V'+str(i),
                                    legendgroup="group_ion",
                                    legendgrouptitle_text="SWEAP_SPAN_A_ELECTRON",
                                    showlegend=False,
                                    line=dict(color='green',width=5)))

        # 连结各个FOV的角
        if i != 12 and i != 25:
            next_ray = np.dot(M_arr_ION,edges[i+1])*length_ray
            # plot.add_trace(go.Scatter3d(x=[spana_electron_center[0]+tmp_ray[0],spana_electron_center[0]+next_ray[0]],
            #                             y=[spana_electron_center[1]+tmp_ray[1],spana_electron_center[1]+next_ray[1]],
            #                             z=[spana_electron_center[2]+tmp_ray[2],spana_electron_center[2]+next_ray[2]],
            #                             mode = 'lines',
            #                             showlegend=False,
            #                             line=dict(color='green')))
            tmp_trace = plot_arc(spana_electron_center,R,spana_electron_center+tmp_ray,spana_electron_center+next_ray,longer_arc=False)
            tmp_trace.update(line=dict(color='green'))
            plot.add_trace(tmp_trace)

        if i < 13:
            next_ray = np.dot(M_arr_ION,edges[25-i])*length_ray
            # plot.add_trace(go.Scatter3d(x=[spana_electron_center[0]+tmp_ray[0],spana_electron_center[0]+next_ray[0]],
            #                             y=[spana_electron_center[1]+tmp_ray[1],spana_electron_center[1]+next_ray[1]],
            #                             z=[spana_electron_center[2]+tmp_ray[2],spana_electron_center[2]+next_ray[2]],
            #                             mode = 'lines',
            #                             showlegend=False,
            #                             line=dict(color='green')))
            tmp_trace = plot_arc(spana_electron_center,R,spana_electron_center+tmp_ray,spana_electron_center+next_ray,longer_arc=False)
            tmp_trace.update(line=dict(color='green'))
            plot.add_trace(tmp_trace)




    plot.add_trace(go.Scatter3d(x=[spana_electron_center[0]],y=[spana_electron_center[1]],z=[spana_electron_center[2]],
                                mode='markers',
                                name='SWEAP_SPAN_A_ELECTRON',
                                marker=dict(size=5,
                                            color='green',
                                            symbol='diamond')))
    plot.update_layout(scene_aspectmode='data',)
    py.plot(plot,filename='SPAN_A_Electron.html')

def plot_arc(center,R,pos_A,pos_B,longer_arc=False,n=50):
    p = np.linspace(0,1,n)
    OA = np.array(pos_A-center)
    OB = np.array(pos_B-center)
    print(OA)
    print(OB)
    OP = np.array([(1-p[i])*OA+p[i]*OB for i in range(n)])
    print(OP.shape)
    print(OP[0])
    print(R)
    print(OA)
    OP = np.array([R*OP[i]/np.linalg.norm(OP[i],2) for i in range(n)])
    if longer_arc:
        OP = -OP
    pos_P = OP+center
    print(pos_P.shape)
    print(pos_P[0])
    print(pos_A)
    print(pos_B)
    arc_trace = go.Scatter3d(x=pos_P[:,0],y=pos_P[:,1],z=pos_P[:,2],
                             mode='lines')
    return arc_trace



if __name__ == '__main__':
    plot_span_a_electron([0,0,0],0,datetime(2021,3,14)) # 输入dt是sxform要求的，对结果无影响（SPP_SPACECRAFT和SWEAP坐标系之间的转换关系不变）
