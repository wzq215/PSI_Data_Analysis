"""
@Filename: Visualization_HCS_PSP_MFL.py
@Aim: plot html file and png file including HCS(SC&IH), PSP and the planets.
@input: time, coordinate
@output：an html file and a png file.
@Author: Ziqi Wu
@Date of Last Change: 2022-04-05
"""
import os
import sys

import imageio
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as py

import sunpy.coordinates.sun as sun
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliocentricInertial
from sunpy.coordinates import HeliographicCarrington

from obtain_mag_line import plot_mag_lines
from Plot_HCS import plot_HCS_SCIH, plot_HCS_rho_SCIH
from Plot_Spacecraft import add_texture
from Plot_Earth import topo_spheres
from Trace_PSI_data import PSI_trace, trace_psp_in_PSI

# from plot_body_positions import xyz2rtp_in_Carrington

import spiceypy as spice

AU = 1.49597870700e8  # km
Rs = 696300  # km


def xyz2rtp_in_Carrington(xyz_carrington,for_psi = False):
    """
    Convert (x,y,z) to (r,t,p) in Carrington Coordination System.
        (x,y,z) follows the definition of SPP_HG in SPICE kernel.
        (r,lon,lat) is (x,y,z) converted to heliographic lon/lat, where lon \in [0,2pi], lat \in [-pi/2,pi/2] .
    :param xyz_carrington:
    :return:
    """
    r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)

    lon_carrington = np.arcsin(xyz_carrington[1]/np.sqrt(xyz_carrington[0]**2+xyz_carrington[1]**2))
    if xyz_carrington[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2*np.pi

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi/2-lat_carrington
    return r_carrington, lon_carrington, lat_carrington



def clon2pos(clon, rotation=0):
    """
    Aim: Convert the carrington longitude to a position vector
    :param clon: A float (deg). Carrington longitude.
    :param rotation: A float (deg), 0 by default . Angle between the target coordinate and the Carrington Coordinate.
    :return: position: A vector [x (Rs), y (Rs), z (Rs)]. Position vector of the object in the target coordinate.
    """
    lon = clon + rotation
    pos = np.array([np.cos(np.deg2rad(lon)), np.sin(np.deg2rad(lon)), 0]) * AU / Rs
    return pos

def plot_PSP_orbit(dts):
    import pandas as pd
    times = spice.datetime2et(dts)
    psp_pos, _ = spice.spkpos('SPP', times, 'IAU_SUN', 'NONE', 'SUN')  # km
    psp_pos = psp_pos.T / Rs
    psp_pos = {'x': psp_pos[0], 'y': psp_pos[1], 'z': psp_pos[2]}
    psp_pos = pd.DataFrame(data=psp_pos)
    trace = go.Scatter3d(x=psp_pos['x'], y=psp_pos['y'], z=psp_pos['z'],
                 mode='lines',
                 line=dict(color='skyblue',
                           width=5),
                 )
    return trace


def plot_one_frame(dt, crid, plot, psp_time_str, coord='HEEQ', plot_planet=False,
                   plot_planet_trace=False, plot_rho_color=False):
    """
    :param dt: A datetime.
    :param crid: An int or float. Carrington Rotation Number.
    :param plot: Figure object of plotly.
    :param S1_lon: An int. Lontitude of spacecraft 1 in HEEQ.
    :param S2_lon: An int. Lontitude of spacecraft 2 in HEEQ.
    :param S3_lon: An int. Lontitude of spacecraft 3 in HEEQ.
    :param coord: A string ('HEEQ' for default.). String defining the coordinates. Valid input: 'HEEQ' or 'HCI' or 'Carrington'.
    :param plot_planet: A bool value. Whether plot planets or not.
    :param plot_planet_trace: A bool value. Whether plot magnetic field lines traced for planets or not.
    :param plot_rho_color: A bool value. Whether plot HCS colored by logged electron density distribution
    :return:
    """

    # Coordinate Transformation by Earth's longitude
    earth = get_body_heliographic_stonyhurst('earth', dt, include_velocity=False)

    earth_carrington = SkyCoord(earth).transform_to(HeliographicCarrington(observer='earth'))
    earth_hci = SkyCoord(earth).transform_to(HeliocentricInertial())

    lon_carrington = earth_carrington.lon.to('rad').value
    lon_hci = earth_hci.lon.to('rad').value
    lon_heeq = earth.lon.to('rad').value
    rot_angle = np.rad2deg(2 * np.pi - lon_carrington)

    if coord == 'HEEQ':
        rotation = - np.rad2deg(lon_carrington + lon_heeq)
        print(rotation)
    elif coord == 'Carrington':
        rotation = 0
    elif coord == 'HCI':
        rotation = - np.rad2deg(lon_hci + lon_carrington)
    else:
        print('Wrong Coordination System! (only available for HEEQ/HCI/Carrington)')
        return 0

    # plot HCS
    if plot_rho_color:
        trace1, trace2 = plot_HCS_rho_SCIH(crid, rotation=rotation)
    elif ~plot_rho_color:
        trace1, trace2 = plot_HCS_SCIH(crid, rotation=rotation)

    plot.add_trace(trace1)
    plot.add_trace(trace2)

    # plot Spacecrafts
    psp_et = spice.datetime2et(datetime.strptime(psp_time_str,'%Y%m%d'))
    # r,clon,lat = xyz2rtp_in_Carrington(psp_pos)
    # clon = np.rad2deg(clon)

    if coord == 'HEEQ':
        psp_pos, _ = spice.spkpos('SPP', psp_et, 'SPP_HEEQ', 'NONE', 'SUN')  # km
        psp_pos = np.array(psp_pos.T / Rs)
    elif coord == 'Carrington':
        psp_pos, _ = spice.spkpos('SPP', psp_et, 'IAU_SUN', 'NONE', 'SUN')  # km
        psp_pos = np.array(psp_pos.T / Rs)
    elif coord == 'HCI':
        psp_pos, _ = spice.spkpos('SPP', psp_et, 'SPP_HCI', 'NONE', 'SUN')  # km
        psp_pos = np.array(psp_pos.T / Rs)

    # r,lon,lat = xyz2rtp_in_Carrington(psp_pos)
    lon = np.rad2deg(np.arccos(psp_pos[0]/np.sqrt(psp_pos[1]**2+psp_pos[0]**2)))
    print(lon)
    if psp_pos[1]<0:
        lon = 360-lon
    # print(clon+rot_angle)
    from datetime import timedelta
    dt_beg = datetime(2020,1,20)
    dt_end = datetime(2020,2,10)
    dt_resolution = timedelta(hours=4)
    dts = [dt_beg + n * dt_resolution for n in range((dt_end - dt_beg) // dt_resolution)]
    plot.add_trace(add_texture(psp_pos,lon+180,scale=20))
    plot.add_trace(plot_PSP_orbit(dts))
    psp_trace = PSI_trace(psp_pos,crid)
    if plot_planet_trace:
        # try:
        # plot.add_trace(
        #     plot_mag_lines(crid, [r, np.pi / 2 - lat, lon], step=3, maxnum=100, rotation=0))
        plot.add_trace(
            go.Scatter3d(x=psp_trace[0].points[:, 0], y=psp_trace[0].points[:, 1], z=psp_trace[0].points[:, 2],
                         mode='markers',
                         marker=dict(size=1),

                         line=dict(color='skyblue',
                                   width=.1),
                         )
        )
        try:
            plot.add_trace(
                go.Scatter3d(x=psp_trace[1].points[:, 0], y=psp_trace[1].points[:, 1], z=psp_trace[1].points[:, 2],
                             mode='markers',
                             marker=dict(size=1),
                             line=dict(color='skyblue',
                                       width=.1),
                             )
            )
        except:
            print('unable to plot trace for psp')


    plot_everywhere_trace = False
    if plot_everywhere_trace:
        lon = np.linspace(0.,2*np.pi,6)
        lat = np.linspace(np.pi/3,np.pi*2/3,3)
        # lat = np.array([np.pi/2])
        # lon = np.array([np.pi/3])
        for ilon in lon:
            for ilat in lat:
                print('(lat, lon)=',np.rad2deg(ilat),np.rad2deg(ilon))
                try:
                    plot.add_trace(plot_mag_lines(crid, [20, ilat,ilon], step_type='var', maxnum=200))
                    print('--succeed--')
                except Exception as e:

                    # print(e.args)
                    # print(str(e))
                    print(repr(e))


    # if plot_everywhere_trace:
    #     # try:
    #     plot.add_trace(
    #         plot_mag_lines(crid, [30., np.pi / 2 - 0., 0.], step_type='var', maxnum=100, rotation=0))
    #     # except:
    #     #     print('unable to plot trace for ', this_planet)

    # sunim = Image.open('data/euvi_aia304_2012_carrington_print.jpeg')
    # sunimgray = ImageOps.grayscale(sunim)
    # sundata = np.array(sunimgray)
    # theta = np.linspace(0, 2 * np.pi, sundata.shape[1])
    # phi = np.linspace(0, np.pi, sundata.shape[0])
    from ps_read_hdf_3d import ps_read_hdf_3d
    data_sc = ps_read_hdf_3d(crid, 'corona', 'br002', periodicDim=3)
    r_br = np.array(data_sc['scales1'])  # 201 in Rs, distance from sun
    t_br = np.array(data_sc['scales2'])  # 150 in rad, latitude
    p_br = np.array(data_sc['scales3'])  # 256 in rad, Carrington longitude
    br = np.array(data_sc['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    br = br * 2.205e5  # nT
    sundata = np.squeeze(br[:,:,0]).T
    # plt.figure()
    # plt.pcolormesh(sundata)
    # plt.colorbar()
    # plt.show()
    # print(sundata.shape)
    theta = p_br
    phi = t_br

    tt, pp = np.meshgrid(theta, phi)
    r = 8
    x0 = r * np.cos(tt) * np.sin(pp)
    y0 = r * np.sin(tt) * np.sin(pp)
    z0 = r * np.cos(pp)

    plot.add_trace(go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='rdbu', opacity=1,showscale=False,cmax=3e5,cmin=-3e5))
    # plot.add_trace(go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='solar', opacity=1, showscale=False))

    # plot planets
    if plot_planet:
        planet_list = ['earth', 'venus', 'mercury']
    else:
        planet_list = ['earth']
    #
    #
    obstime = dt

    planet_coord = [get_body_heliographic_stonyhurst(
        this_planet, time=obstime) for this_planet in planet_list]

    for this_planet, this_coord in zip(planet_list, planet_coord):
        lon = this_coord.lon.to('rad').value
        lat = this_coord.lat.to('rad').value
        r = this_coord.radius.value * AU / Rs
        if coord == 'HEEQ':
            lon = lon
        elif coord == 'Carrington':
            new_coord = SkyCoord(this_coord).transform_to(HeliographicCarrington(observer='earth'))
            lon = new_coord.lon.to('rad').value
            # lon = lon - lon_carrington - lon_heeq
        elif coord == 'HCI':
            # new_coord = SkyCoord(this_coord).transform_to(HeliocentricInertial())
            # lon = new_coord.lon.to('rad').value
            lon = lon - lon_hci - lon_heeq

        pos = np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]) * r
        plot.add_trace(topo_spheres(6, pos, planet=this_planet))
        print(this_planet)
        print(r)
        # print(lat)
        this_trace = PSI_trace(pos,crid)
        print(this_trace)
        # this_trace_points = this_trace

        if plot_planet_trace:
            # try:
                # plot.add_trace(
                #     plot_mag_lines(crid, [r, np.pi / 2 - lat, lon], step=3, maxnum=100, rotation=0))
            plot.add_trace(
                    go.Scatter3d(x=this_trace[0].points[:,0], y=this_trace[0].points[:,1], z=this_trace[0].points[:,2],
                 mode='markers',
                                 marker=dict(size=1),

                 line=dict(color='white',
                           width=.1),
                 )
                )
            try:
                plot.add_trace(
                    go.Scatter3d(x=this_trace[1].points[:, 0], y=this_trace[1].points[:, 1], z=this_trace[1].points[:, 2],
                                 mode='markers',
                                 marker=dict(size=1),
                                 line=dict(color='white',
                                           width=.1),
                                 )
                )
            except:
                print('unable to plot trace for ', this_planet)


    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-1.5, z=1.5)
    )

    plot.update_layout(
        scene=dict(
            xaxis_title='X (Rs)',
            yaxis_title='Y (Rs)',
            zaxis_title='Z (Rs)',
            xaxis_range=[-250, 250],
            yaxis_range=[-250, 250],
            zaxis_range=[-250, 250],
        ),
        scene_camera=camera,
        title=dict(
            text="CR" + str(crid) + ' ' + coord + ' ' + 'Coordinate',
            font_size=30,
            y=0.9, x=0.5,
            xanchor='center',
            yanchor='top'),
        template='plotly_dark',
        showlegend=False,
        scene_aspectratio=dict(x=1, y=1, z=1),
    )
    plot.update_scenes(xaxis_tickfont_size=15,
                       yaxis_tickfont_size=15,
                       zaxis_tickfont_size=15,
                       xaxis_title_font_size=25,
                       yaxis_title_font_size=25,
                       zaxis_title_font_size=25,

                       )


def plot_static_images(yy, mm, dd, psp_time_str, type='jpg', coord='HEEQ'):
    """
    :param yy: An int. Year of the target plot.
    :param mm: An int. Month of the target plot.
    :param dd: An int. Day of the target plot.
    :param s1_lon: An int. Lontitude of spacecraft 1 in HEEQ.
    :param s2_lon: An int. Lontitude of spacecraft 2 in HEEQ.
    :param s3_lon: An int. Lontitude of spacecraft 3 in HEEQ.
    :param type: A string. 'jpg' or 'html'
    :param coord: A string. 'HEEQ' or 'HCI' or 'Carrington'
    """
    dt = datetime(yy, mm, dd)
    cr = sun.carrington_rotation_number(t=dt)
    t = dt.strftime('%Y%m%d')
    crid = int(np.floor(cr))
    print(yy, mm, dd)
    print('Carrington Rotation Number:', crid)
    plot = go.Figure()
    plot_one_frame(dt, crid, plot, psp_time_str, coord=coord, plot_planet=True,
                   plot_planet_trace=True,
                   plot_rho_color=True)
    if type == 'html':
        py.plot(plot, filename='../../搁置的程序/HCS-PSP可视化/plot_PSP_HCS_Planets/HCS_PSP_PLANET.html')
    elif type == 'jpg':
        plot.write_image('HCS_PSP_PLANET.' + type, height=1200, width=1080)


if __name__ == '__main__':
    dt = datetime(2020,1,31,)
    time_str = dt.strftime('%Y%m%d')
    plot_static_images(dt.year, dt.month, dt.day, time_str, type='html', coord='Carrington')
