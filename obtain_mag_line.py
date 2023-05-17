"""
@Filename: obtain_mag_line.py
@Aim: read HDF files from PSI (br/bt/bp in corona/helio) and compute the magnetic field line traced back from an initial point to the Sun
@Author: Chuanpeng Hou, Ziqi Wu.
@Date of Last Change: 2022-01-24
"""

import numpy as np
import plotly.graph_objects as go
import plotly.offline as py

from RBF_B import magnetic_RBF_inter
from ps_read_hdf_3d import ps_read_hdf_3d


def get_xyz(r, theta, phi):
    """
    :param r: r in Spherical Cordinates, unit: Rs
    :param theta: theta in Spherical Coordinates, unit: rad
    :param phi: phi in Spherical Coordinates, unit: rad
    :return: [x,y,z] in Cartesian Coordinates, unit: [Rs,Rs,Rs]
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    xyz = np.array([x, y, z])
    return xyz


def mag_from_rtp_to_xyz(Br, Bt, Bp, theta, phi):
    """
    :aim: obtain magnetic field vector in Cartesian Coordinates
    :return: Bxyz: magnetic field vector in Cartesian Coordinates
    """
    Bx = Br * np.sin(theta) * np.cos(phi) + Bt * np.cos(theta) * np.cos(phi) - Bp * np.sin(phi)
    By = Br * np.sin(theta) * np.sin(phi) + Bt * np.cos(theta) * np.sin(phi) + Bp * np.cos(phi)
    Bz = Br * np.cos(theta) - Bt * np.sin(theta)
    Bxyz = np.array([Bx, By, Bz])
    return Bxyz


def obtain_step_of_single_point(cr, destination_point):
    dict_corona, Br_corona, Bt_corona, Bp_corona, \
    dict_helio, Br_helio, Bt_helio, Bp_helio = obtain_Brtp_and_grid_coordinate(cr)
    r_index_of_nearest_point = np.argmin(np.abs(dict_helio['scales1'] - destination_point[0]))
    step = np.diff(np.array(dict_helio['scales1']))[r_index_of_nearest_point]
    if r_index_of_nearest_point == 0:
        r_index_of_nearest_point = np.argmin(np.abs(dict_corona['scales1'] - destination_point[0]))
        step = np.diff(np.array(dict_helio['scales1']))[r_index_of_nearest_point - 1]
    return step


def obtain_B_of_single_point(cr, destination_point):
    """
    :param cr:  Carrington cycle, such as 2246. single value
    :param destination_point: target point vector in Spherical Coordinates. shape: 1D
    :return: magnetic field vector at target point in Spherical Coordinates. shape: 1D
    """
    dict_corona, Br_corona, Bt_corona, Bp_corona, \
    dict_helio, Br_helio, Bt_helio, Bp_helio = obtain_Brtp_and_grid_coordinate(cr)
    # destination_point = np.array([r_i,theta_i,phi_i])
    r_index_of_nearest_point = np.argmin(np.abs(dict_helio['scales1'] - destination_point[0]))
    theta_index_of_nearest_point = np.argmin(np.abs(dict_helio['scales2'] - destination_point[1]))
    phi_index_of_nearest_point = np.argmin(np.abs(dict_helio['scales3'] - destination_point[2]))
    # print(r_index_of_nearest_point)
    if r_index_of_nearest_point == 0:
        r_index_of_nearest_point = np.argmin(np.abs(dict_corona['scales1'] - destination_point[0]))
        if r_index_of_nearest_point == 0:
            B = 42
            # print('Current point is beyond the inner boundary!')
            return B
        if r_index_of_nearest_point == len(dict_corona['scales1']) - 1:
            if (np.abs(dict_corona['scales1'][-1] - destination_point[0])) > (
                    np.abs(dict_helio['scales1'][0] - destination_point[0])):  # ! Revised
                r_index_of_nearest_point = 0
                # print('Current point is near the internal boundary of helio!')
            else:
                theta_index_of_nearest_point = np.argmin(np.abs(dict_corona['scales2'] - destination_point[1]))
                phi_index_of_nearest_point = np.argmin(np.abs(dict_corona['scales3'] - destination_point[2]))
                # print('Current point is near the external boundary of corona!')
                pass
    # print(r_index_of_nearest_point)
    nv = 27
    m = 26
    B = magnetic_RBF_inter(dict_corona['scales1'], dict_corona['scales2'], dict_corona['scales3'],
                           Br_corona, Bt_corona, Bp_corona,
                           dict_helio['scales1'], dict_helio['scales2'], dict_helio['scales3'],
                           Br_helio, Bt_helio, Bp_helio,
                           destination_point[0], destination_point[1], destination_point[2],
                           nv, m,
                           r_index_of_nearest_point,
                           theta_index_of_nearest_point,
                           phi_index_of_nearest_point)
    return B


def obtain_Brtp_and_grid_coordinate(cr):
    """
    :aim: load magnetic field vector data for corona and helio
    :param cr: Carrington cycle, such as 2246. single value
    :return:  dict_corona,Br_corona,Bt_corona,Bp_corona,dict_helio,Br_helio,Bt_helio,Bp_helio
    """
    dict_corona = ps_read_hdf_3d(cr, 'corona', 'br002', 3)
    dict_helio = ps_read_hdf_3d(cr, 'helio', 'br002', 3)
    br_p_corona = dict_corona['scales3']
    br_p_helio = dict_helio['scales3']
    Br_corona = dict_corona['datas']
    Br_helio = dict_helio['datas']

    dict_corona = ps_read_hdf_3d(cr, 'corona', 'bt002', 3)
    dict_helio = ps_read_hdf_3d(cr, 'helio', 'bt002', 3)
    bt_r_corona = dict_corona['scales1']
    bt_r_helio = dict_helio['scales1']
    Bt_corona = dict_corona['datas']
    Bt_helio = dict_helio['datas']

    dict_corona = ps_read_hdf_3d(cr, 'corona', 'bp002', 3)
    dict_helio = ps_read_hdf_3d(cr, 'helio', 'bp002', 3)
    bp_t_corona = dict_corona['scales2']
    bp_t_helio = dict_helio['scales2']
    Bp_corona = dict_corona['datas']
    Bp_helio = dict_helio['datas']

    rg_corona = bt_r_corona
    tg_corona = bp_t_corona
    pg_corona = br_p_corona
    dict_g_corona = {'scales1': rg_corona, 'scales2': tg_corona, 'scales3': pg_corona}

    rg_helio = bt_r_helio
    tg_helio = bp_t_helio
    pg_helio = br_p_helio
    dict_g_helio = {'scales1': rg_helio, 'scales2': tg_helio, 'scales3': pg_helio}

    br_g_corona = (Br_corona[:, :, 1:] + Br_corona[:, :, :-1]) / 2.
    bt_g_corona = (Bt_corona[:, 1:, :] + Bt_corona[:, :-1, :]) / 2.
    bp_g_corona = (Bp_corona[1:, :, :] + Bp_corona[:-1, :, :]) / 2.

    br_g_helio = (Br_helio[:, :, 1:] + Br_helio[:, :, :-1]) / 2.
    bt_g_helio = (Bt_helio[:, :, 1:] + Bt_helio[:, :, :-1]) / 2.
    bp_g_helio = (Bp_helio[:, :, 1:] + Bp_helio[:, :, :-1]) / 2.

    return dict_g_corona, br_g_corona, bt_g_corona, bp_g_corona, dict_g_helio, br_g_helio, bt_g_helio, bp_g_helio

def streamtubeplot_plotly(cr):
    dict_g_corona, br_g_corona, bt_g_corona, bp_g_corona, dict_g_helio, br_g_helio, bt_g_helio, bp_g_helio=obtain_Brtp_and_grid_coordinate(cr)
    rg_corona = dict_g_corona['scales1']
    tg_corona = dict_g_corona['scales2']
    pg_corona = dict_g_corona['scales3']
    tgv_sc, pgv_sc, rgv_sc = np.meshgrid(tg_corona, pg_corona, rg_corona, indexing='xy')
    xgv = rgv_sc * np.sin(tgv_sc) * np.cos(pgv_sc)
    ygv = rgv_sc * np.sin(tgv_sc) * np.sin(pgv_sc)
    zgv = rgv_sc * np.cos(tgv_sc)
    Bxg = br_g_corona * np.sin(tgv_sc) * np.cos(pgv_sc) + bt_g_corona * np.cos(tgv_sc) * np.cos(pgv_sc) - bp_g_corona * np.sin(pgv_sc)
    Byg = br_g_corona * np.sin(tgv_sc) * np.sin(pgv_sc) + bt_g_corona * np.cos(tgv_sc) * np.sin(pgv_sc) + bp_g_corona * np.cos(pgv_sc)
    Bzg = br_g_corona * np.cos(tgv_sc) - bt_g_corona * np.sin(tgv_sc)

    # plot = go.Figure()
    # plot.add_trace(go.Streamtube(x=xgv.flatten(),y=ygv.flatten(),z=zgv.flatten(),u=Bxg.flatten(),v=Byg.flatten(),w=Bzg.flatten(),
    #                              starts=dict(x=[25.],y=[0.],z=[0.]),colorscale = 'Portland'))
    # py.plot(plot)
    # from streamtracer import StreamTracer, VectorGrid
    # nsteps = 10000
    # step_size = 0.1
    # tracer = StreamTracer(nsteps, step_size)
    #
    # field = np.zeros((Bxg.shape[0],Bxg.shape[1],Bxg.shape[2],3))
    # field[:,:,:,0] = Bxg
    # field[:,:,:,1] = Byg
    # field[:,:,:,2] = Bzg

    # grid_coords = np.zeros((xgv.size,3))
    # grid_coords[:,0] = xgv.flatten()
    # grid_coords[:,1] = ygv.flatten()
    # grid_coords[:,2] = zgv.flatten()

    # grid = VectorGrid(field, grid_coords=[xgv.flatten(),ygv.flatten(),zgv.flatten()])
    # tracer.trace(seeds, grid)
    # print(tracer)

    import pyvista
    mesh = pyvista.StructuredGrid(xgv, ygv, zgv)
    vectors = np.empty((mesh.n_points,3))
    vectors[:,0] = Bxg.ravel()
    vectors[:,1] = Byg.ravel()
    vectors[:,2] = Bzg.ravel()
    mesh['vectors'] = vectors
    stream, src = mesh.streamlines(
        'vectors', return_source=True, terminal_speed=0.0, n_points=100, source_radius=20.
    )
    cpos = [(1.2, 1.2, 1.2), (-0.0, -0.0, -0.0), (0.0, 0.0, 1.0)]
    stream.tube(radius=0.15).plot(cpos=cpos)
    # mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
    # seeds = np.array([[0, 0, 25]])




def obtain_position_of_next_point(current_pos, step, current_B):
    """
    :param current_pos: current position in Spherical Coordinates. shape: 1D
    :param step: step length unit: Rs. single value
    :param current_B:current magnetic field vector in Spherical Coordinates. shape: 1D
    :return: magnetic field vector at next position in Spherical Coordinates. shape: 1D
    """
    current_pos_in_xyz = get_xyz(current_pos[0], current_pos[1], current_pos[2])
    current_B_in_xyz = mag_from_rtp_to_xyz(current_B[0], current_B[1], current_B[2],
                                           current_pos[1], current_pos[2])
    current_B_amp = np.sqrt(current_B_in_xyz[0] ** 2 + current_B_in_xyz[1] ** 2 + current_B_in_xyz[2] ** 2)

    if current_B[0] > 0:
        next_pos_in_xyz = current_pos_in_xyz - step * current_B_in_xyz / current_B_amp
        next_pos = get_rtp(next_pos_in_xyz)
    else:
        next_pos_in_xyz = current_pos_in_xyz + step * current_B_in_xyz / current_B_amp
        next_pos = get_rtp(next_pos_in_xyz)
    return next_pos


def get_rtp(old_data_in_xyz):
    """
    :param old_data_in_xyz: [x,y,z] in Cartesian Cordinates, unit: Rs
    :return: data_in_rtf: [r,theta,phi] in Spherical Cordinates, unit: [Rs,rad,rad]
    """
    x = old_data_in_xyz[0]
    y = old_data_in_xyz[1]
    z = old_data_in_xyz[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    if x > 0 and y > 0:
        phi = np.arcsin(y / np.sqrt(x ** 2 + y ** 2))
    elif x < 0 and y > 0:
        phi = np.arccos(x / np.sqrt(x ** 2 + y ** 2))
    elif x < 0 and y < 0:
        phi = np.pi - np.arcsin(y / np.sqrt(x ** 2 + y ** 2))
    else:
        phi = 2 * np.pi + np.arcsin(y / np.sqrt(x ** 2 + y ** 2))
    data_in_rtf = np.array([r, theta, phi])
    return data_in_rtf


def obtain_mag_line(cr, initial_pos, step_type, maxnum=100):
    """

    :param cr: Carrington Rotation Number
    :param initial_pos: [r(Rs),lat(rad),lon(rad)] in Carrington Coordinates
    :param step: dx(Rs)
    :param maxnum: maximum steps
    :return: pos_r_of_mag_line (Rs), pos_t_of_mag_line (rad), pos_p_of_mag_line (rad),
    Br_of_mag_line (CU), Bt_of_mag_line (CU), Bp_of_mag_line (CU): position vector and magnetic field vector along a magnetic field line
    """
    current_B = obtain_B_of_single_point(cr, initial_pos)
    if step_type != 'var':
        step = step_type
        print(step)
    current_pos = initial_pos
    pos_r_of_mag_line = np.array([current_pos[0]])  # unit:Rs
    pos_t_of_mag_line = np.array([current_pos[1]])  # unit:rad
    pos_p_of_mag_line = np.array([current_pos[2]])  # unit:rad
    Br_of_mag_line = np.array([current_B[0]])  # unit:cu
    Bt_of_mag_line = np.array([current_B[1]])  # unit:cu
    Bp_of_mag_line = np.array([current_B[2]])  # unit:cu
    num = 0

    dict_corona, Br_corona, Bt_corona, Bp_corona, \
    dict_helio, Br_helio, Bt_helio, Bp_helio = obtain_Brtp_and_grid_coordinate(cr)
    while not current_B[0] == 42 and num <= maxnum:  # 等于42表示到了内边界， num限制步数
        try:
            if step_type == 'var':
                r_index_of_nearest_point = np.argmin(np.abs(dict_helio['scales1'] - current_pos[0]))
                step = np.diff(np.array(dict_helio['scales1']))[r_index_of_nearest_point] * 2.
                if r_index_of_nearest_point == 0:
                    r_index_of_nearest_point = np.argmin(np.abs(dict_corona['scales1'] - current_pos[0]))
                    step = np.diff(np.array(dict_corona['scales1']))[r_index_of_nearest_point] * 2.
            print('step=', step)
            next_pos = obtain_position_of_next_point(current_pos, step, current_B)
            next_B = obtain_B_of_single_point(cr, next_pos)
            pos_r_of_mag_line = np.append(pos_r_of_mag_line, next_pos[0])
            pos_t_of_mag_line = np.append(pos_t_of_mag_line, next_pos[1])
            pos_p_of_mag_line = np.append(pos_p_of_mag_line, next_pos[2])
            Br_of_mag_line = np.append(Br_of_mag_line, next_B[0])
            Bt_of_mag_line = np.append(Bt_of_mag_line, next_B[0])
            Bp_of_mag_line = np.append(Bp_of_mag_line, next_B[0])
            current_pos = next_pos
            current_B = next_B
            # print('step_num = ', num , ', step = ', step ,'Rs')
            # print('current_pos ', current_pos)
            num = num + 1
        except (Exception, BaseException) as e:
            # print('Error occured at step num = ',num,'. Stop tracing.')
            # exstr = traceback.format_exc()
            # print(exstr)
            # print('-----------------------------')
            break
    return pos_r_of_mag_line, pos_t_of_mag_line, pos_p_of_mag_line, Br_of_mag_line, Bt_of_mag_line, Bp_of_mag_line


def plot_mag_lines(crid, pos, step_type=1, maxnum=100, rotation=0):
    """
    :param crid: Carrington Rotation Number
    :param pos: [r(Rs),lat(rad),lon(rad)] in Carrington Coordinates
    :param step: dx(Rs)
    :param maxnum: maximum steps
    :param rotation: rotate around z axis centered at (0,0,0), Anti-clockwise.
    :return:  A trace (go.Scatter3d(mode='lines')) for plotly
    """
    mag_r, mag_t, mag_p, mag_br, mag_bt, mag_bp = obtain_mag_line(crid, pos, step_type, maxnum)
    mag_p += np.deg2rad(rotation)
    mag_x = mag_r * np.cos(mag_p) * np.sin(mag_t)
    mag_y = mag_r * np.sin(mag_p) * np.sin(mag_t)
    mag_z = mag_r * np.cos(mag_t)
    theta_Br = np.arccos(mag_br / np.sqrt(mag_br ** 2 + mag_bp ** 2 + mag_bt ** 2))
    total_B = np.sqrt(mag_br ** 2 + mag_bt ** 2 + mag_bp ** 2)
    print(theta_Br)
    trace = go.Scatter3d(x=mag_x, y=mag_y, z=mag_z,
                         mode='lines',
                         line=dict(color=total_B, colorscale='viridis', cmin=10, cmax=500,
                                   width=5, showscale=True)
                         )
    return trace


if __name__ == '__main__':

    streamtubeplot_plotly(2243)
    quit()
    cr = 2243
    r_i = 214.8
    theta_i = np.pi / 2
    phi_i = np.pi / 3
    initial_pos = np.array([r_i, theta_i, phi_i])
    step = 2  # unit: Rs

    # pos_r_of_mag_line, pos_t_of_mag_line, pos_p_of_mag_line, Br_of_mag_line, Bt_of_mag_line, Bp_of_mag_line = \
    # obtain_mag_line(cr, initial_pos ,step)  # unit of mag: CU
    #
    # plt.plot(pos_r_of_mag_line,Br_of_mag_line)
    # plt.show()
    plot = go.Figure()
    lon = np.linspace(0., 2 * np.pi, 3)
    lat = np.linspace(np.pi / 3, np.pi * 2 / 3, 3)
    # lat = np.array([np.pi/2])
    # lon = np.array([np.pi/3])
    for ilon in lon:
        for ilat in lat:
            print('(lat, lon)=', np.rad2deg(ilat), np.rad2deg(ilon))
            try:
                plot.add_trace(plot_mag_lines(cr, [25, ilat, ilon], step_type='var', maxnum=100))
                print('--succeed--')
            except Exception as e:

                # print(e.args)
                # print(str(e))
                print(repr(e))
                print('--fail--')
    plot.update_layout(
        scene=dict(
            xaxis_title='X (Rs)',
            yaxis_title='Y (Rs)',
            zaxis_title='Z (Rs)',
            xaxis_range=[-30, 30],
            yaxis_range=[-30, 30],
            zaxis_range=[-30, 30],
        ),
        showlegend=False,
        scene_aspectratio=dict(x=1, y=1, z=1),
    )
    py.plot(plot)
