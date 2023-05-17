import numpy as np
from datetime import datetime,timedelta

import sunpy.coordinates.sun as sun
from matplotlib import pyplot as plt
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliocentricInertial
from sunpy.coordinates import HeliographicCarrington
from ps_read_hdf_3d import ps_read_hdf_3d
import pandas as pd
import furnsh_kernels_psp
Rs = 696300
AU = 1.5e8
unit_lst = {'l':Rs,
            'vr':481.3711, 'vt':481.3711, 'vp': 481.3711, # km/s
            'ne':1e14,'rho':1e8,'p':3.875717e-2,'T':2.807067e7,
            'br':2.2068908e5,'bt':2.2068908e5,'bp':2.2068908e5, # nT
            'j':2.267e4}

def sample_in_psi(epoch,position,param_lst,lat2psilat=True):
    sample={param:np.zeros_like(epoch) for param in param_lst}
    if lat2psilat:
        position['lat'] = np.pi/2-position['lat']

    crid_beg = int(sun.carrington_rotation_number(epoch[0]))
    crid_end = int(sun.carrington_rotation_number(epoch[-1]))

    for crid in range(crid_beg,crid_end+1):
        subepochbin = (epoch>=sun.carrington_rotation_time(crid)) & (epoch<sun.carrington_rotation_time(crid+1))
        subepochbin_scbin = subepochbin & (position['r']<=30)
        subepochbin_ihbin = subepochbin & (position['r']>30)

        if sum(subepochbin_ihbin) > 0:
            for param in param_lst:
                data_ih = ps_read_hdf_3d(crid, 'helio', param+'002', periodicDim=3)
                r = np.array(data_ih['scales1'])
                t = np.array(data_ih['scales2'])
                p = np.array(data_ih['scales3'])
                datas = np.array(data_ih['datas'])
                datas = datas * unit_lst[param]

                r_inds = list(map(lambda x: np.argmin(abs(x-r)),np.array(position['r'])))
                t_inds = list(map(lambda x: np.argmin(abs(x-t)),np.array(position['lat'])))
                p_inds = list(map(lambda x: np.argmin(abs(x-p)),np.array(position['lon'])))

                sample[param][subepochbin_ihbin] = datas[p_inds,t_inds,r_inds][subepochbin_ihbin]
        if sum(subepochbin_scbin)>0:
            for param in param_lst:
                data_sc = ps_read_hdf_3d(crid, 'corona', param+'002', periodicDim=3)
                r = np.array(data_sc['scales1'])  # 201 in Rs, distance from sun
                t = np.array(data_sc['scales2'])  # 150 in rad, latitude
                p = np.array(data_sc['scales3'])  # 256 in rad, Carrington longitude
                datas = np.array(data_sc['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
                datas = datas * unit_lst[param]  #

                r_inds = list(map(lambda x: np.argmin(abs(x - r)), position['r']))
                t_inds = list(map(lambda x: np.argmin(abs(x - t)), position['lat']))
                p_inds = list(map(lambda x: np.argmin(abs(x - p)), position['lon']))

                sample[param][subepochbin_scbin] = datas[p_inds, t_inds, r_inds][subepochbin_ihbin]
    return sample

def polar_orbit(epoch,coord='HG'):

    T_orbit = timedelta(days=360)
    T_sun =(sun.carrington_rotation_time(2001)-sun.carrington_rotation_time(2000)).value
    T_sun = timedelta(days=T_sun)

    R = AU/Rs
    t0 = epoch[0]

    earth = get_body_heliographic_stonyhurst('earth', t0, include_velocity=False)
    earth_hci = SkyCoord(earth).transform_to(HeliocentricInertial())
    earth_carrington = SkyCoord(earth).transform_to(HeliographicCarrington(observer='earth'))

    earth_lon_hci = earth_hci.lon.to('rad').value
    earth_lon_carrington =earth_carrington.lon.to('rad').value

    t_orbit = np.array((epoch-t0)/T_orbit,dtype=float)
    t_sun = np.array((epoch-t0)/T_sun,dtype=float)

    if coord == 'HCI':
        r = R+np.zeros_like(t_orbit)
        lon = earth_lon_hci
        lat = (2*np.pi*t_orbit)

        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        lon = lon % (2 * np.pi)
        lat = np.arcsin(np.sin(lat))

    elif coord == 'HG':
        r = R+np.zeros_like(t_orbit)
        lon = (earth_lon_carrington - 2*np.pi*t_sun)
        lat = (2*np.pi*t_orbit)
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        lon = lon % (2 * np.pi)
        lat = np.arcsin(np.sin(lat))



    return {'r':r, 'lon':lon, 'lat':lat, 'x':x, 'y':y, 'z':z, 'coord':coord}


if __name__ == '__main__':

    start_time = datetime(2019,1,1)
    stop_time = datetime(2019,12,31)
    start_time_str = start_time.strftime('%Y-%m-%d')
    stop_time_str = stop_time.strftime('%Y-%m-%d')


    timestep = timedelta(days=1)
    steps = (stop_time - start_time) // timestep + 1
    epoch = np.array([x * timestep + start_time for x in range(steps)])
    positions_HG = polar_orbit(epoch,coord='HG')
    positions_HCI = polar_orbit(epoch,coord='HCI')

    # %%
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions_HG['x'], positions_HG['y'], positions_HG['z'])
    ax.scatter(1, 0, 0, c='red')
    ax.set_xlabel('X (Rs)')
    ax.set_ylabel('Y (Rs)')
    ax.set_zlabel('Z (Rs)')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_aspect('equal')
    plt.title('HG Coordinates')
    plt.show()

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions_HCI['x'], positions_HCI['y'], positions_HCI['z'])
    ax.scatter(1, 0, 0, c='red')
    ax.set_xlabel('X (Rs)')
    ax.set_ylabel('Y (Rs)')
    ax.set_zlabel('Z (Rs)')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_aspect('equal')
    plt.title('HCI Coordinates')
    plt.show()

    plt.figure()
    plt.scatter(np.rad2deg(positions_HG['lon']),np.rad2deg(positions_HG['lat']),c=np.arange(len(epoch)))
    plt.xlabel('Carrington Longitude')
    plt.ylabel('Carrington Latitude')
    plt.colorbar()
    plt.title('Coverage ')
    plt.show()
    # %%
    sample = sample_in_psi(epoch,positions_HG,['br','bt','bp','vr','vt','vp','rho'])
    print(sample)
    # %%
    sample_df = pd.DataFrame(sample)
    sample_df.plot(subplots=True,title='Year '+str(start_time.year))
    # %%








