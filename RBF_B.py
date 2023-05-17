"""
@Filename: RBF_B.py
@Aim: interpolate magnetic field vector based on a radial basis function method, which can keep the magnetic field locally divergence-free.
@input: grid data and position of target point.
@output：magnetic field vector at target point in Spherical Cordinates. shape:[Br, Btheta, Bphi]
@Author: Chuanpeng Hou
@Date of Last Change: 2022-01-24
"""
import numpy as np


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


def RBF_function_partial1(r_support, x, y, z, xj, yj, zj):
    """
    :param r_support: single value which is two times of maximum value of distance between target point and neighbor points.
    :param x: x of neighbor points in Cartesian Coordinates. shape: 1D
    :param y: y of neighbor points in Cartesian Coordinates. shape: 1D
    :param z: z of neighbor points in Cartesian Coordinates. shape: 1D
    :param xj: x of target point in Cartesian Coordinates. single value
    :param yj: y of target point in Cartesian Coordinates. single value
    :param zj: z of target point in Cartesian Coordinates. single value
    :return: partial: the basis function
    """
    alph = 1.0 / r_support ** 2
    temp = np.exp(-alph * ((x - xj) ** 2 + (y - yj) ** 2 + (z - zj) ** 2))
    pxx = temp * (4 * alph ** 2 * (x - xj) ** 2 - 2 * alph)
    pyy = temp * (4 * alph ** 2 * (y - yj) ** 2 - 2 * alph)
    pzz = temp * (4 * alph ** 2 * (z - zj) ** 2 - 2 * alph)
    pxy = 4 * alph ** 2 * (x - xj) * (y - yj) * temp
    pxz = 4 * alph ** 2 * (x - xj) * (z - zj) * temp
    pyz = 4 * alph ** 2 * (y - yj) * (z - zj) * temp

    partial = np.zeros([3, 3])
    partial[0, 0] = - pyy - pzz
    partial[0, 1] = pxy
    partial[0, 2] = pxz
    partial[1, 0] = pxy
    partial[1, 1] = -pxx - pzz
    partial[1, 2] = pyz
    partial[2, 0] = pxz
    partial[2, 1] = pyz
    partial[2, 2] = - pxx - pyy
    return partial


def polynomial_magnetic(x, y, z, m):
    """
    :param x:  x of neighbor points in Cartesian Coordinates. shape: 1D
    :param y:  y of neighbor points in Cartesian Coordinates. shape: 1D
    :param z:  z of neighbor points in Cartesian Coordinates. shape: 1D
    :param m:  the number of polynomial functions. sigle value
    :return: p: polynomial functions
    """
    p = np.zeros(3)
    if m == 0:
        p[0] = 1
    elif m == 1:
        p[1] = 1
    elif m == 2:
        p[2] = 1
    elif m == 3:
        p[0] = y
    elif m == 4:
        p[0] = z
    elif m == 5:
        p[1] = x
    elif m == 6:
        p[1] = z
    elif m == 7:
        p[2] = x
    elif m == 8:
        p[2] = y
    elif m == 9:
        p[0] = x
        p[1] = -y
    elif m == 10:
        p[0] = x
        p[2] = -z
    elif m == 11:
        p[2] = x ** 2
    elif m == 12:
        p[1] = x ** 2
    elif m == 13:
        p[2] = y ** 2
    elif m == 14:
        p[0] = y ** 2
    elif m == 15:
        p[1] = z ** 2
    elif m == 16:
        p[0] = z ** 2
    elif m == 17:
        p[2] = x * y
    elif m == 18:
        p[0] = x ** 2
        p[1] = -2 * x * y
    elif m == 19:
        p[0] = -x ** 2
        p[2] = 2 * x * z
    elif m == 20:
        p[1] = x * z
    elif m == 21:
        p[0] = 2 * x * y
        p[1] = -y ** 2
    elif m == 22:
        p[0] = -2 * x * z
        p[2] = z ** 2
    elif m == 23:
        p[1] = y ** 2
        p[2] = -2 * y * z
    elif m == 24:
        p[0] = y * z
    elif m == 25:
        p[1] = 2 * y * z
        p[2] = -z ** 2
    return p


def svd_solver(A, c):
    """
    :param A: matrix
    :param c: vector
    :return: w: =A^(-1)c, a solution of linear system of equations
    """
    # A*w=c -> w = A^(-1) * c
    A_inv = np.linalg.inv(A)
    w = np.dot(A_inv, c)
    return w


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


def mag_from_xyz_to_rtf(Bx, By, Bz, theta, phi):
    """
    :aim: obtain magnetic field vector in Spherical Cordinates
    :return: Brtp: magnetic field vector in Spherical Cordinates
    """
    Br = Bx * np.sin(theta) * np.cos(phi) + By * np.sin(theta) * np.sin(phi) + Bz * np.cos(phi)
    Bt = Bx * np.cos(theta) * np.cos(phi) + By * np.cos(theta) * np.sin(phi) - Bz * np.sin(phi)
    Bp = - Bx * np.sin(phi) + By * np.cos(phi)
    Brtp = np.array([Br, Bt, Bp])
    return Brtp


def obtain_proper_length_xyz_data(x_data2_old, y_data2_old, z_data2_old, Bx_data2, By_data2, Bz_data2):
    """
    :aim: make sure three components of magnetic field vector possess the same size.
    """
    Br_helio_shape = np.shape(Bx_data2)
    Bt_helio_shape = np.shape(By_data2)
    Bp_helio_shape = np.shape(Bz_data2)
    Br_helio_max_index = np.min([Br_helio_shape[2], Bt_helio_shape[2], Bp_helio_shape[2]])
    Bt_helio_max_index = np.min([Br_helio_shape[1], Bt_helio_shape[1], Bp_helio_shape[1]])
    Bp_helio_max_index = np.min([Br_helio_shape[0], Bt_helio_shape[0], Bp_helio_shape[0]])
    x_data2 = x_data2_old[0:Br_helio_max_index]
    y_data2 = y_data2_old[0:Bt_helio_max_index]
    z_data2 = z_data2_old[0:Bp_helio_max_index]
    return x_data2, y_data2, z_data2


def obtain_proper_nearest_index(j, k, y_data, z_data):
    """
    :aim: tackle the boundaries in theta and phi direction.
    """
    if j > len(y_data) - 1:
        j = 0
    elif j < 0:
        j = len(y_data) - 1
    else:
        pass
    if k > len(z_data) - 1:
        k = 0
    elif k < 0:
        k = len(z_data) - 1
    else:
        pass
    # print('len(y_data,zdata):', len(y_data), len(z_data))
    # print('current_index: ', j, k)
    return j, k


def magnetic_RBF_inter(x_data1_old, y_data1_old, z_data1_old,
                       Bx_data1, By_data1, Bz_data1,
                       x_data2_old, y_data2_old, z_data2_old,
                       Bx_data2, By_data2, Bz_data2,
                       r_i, theta_i, phi_i, nv, m,  # nv: 笛卡尔坐标系，待插值点周围3*3*3=27个点 m=nv-1=26
                       x_index_of_nearest_point,
                       y_index_of_nearest_point,
                       z_index_of_nearest_point):
    """
    :param x_data1_old: r of coronal data in Spherical Cordinates. shape: 1D
    :param y_data1_old: theta of coronal data in Spherical Cordinates. shape: 1D
    :param z_data1_old: phi of coronal data in Spherical Cordinates. shape: 1D
    :param Bx_data1: B_r of coronal data in Spherical Cordinates. shape: 3D
    :param By_data1: B_theta of coronal data in Spherical Cordinates. shape: 3D
    :param Bz_data1: B_phi of coronal data in Spherical Cordinates. shape: 3D
    :param x_data2_old: r of helio's data in Spherical Cordinates. shape: 1D
    :param y_data2_old: theta of helio's data in Spherical Cordinates. shape: 1D
    :param z_data2_old: phi of helio's data in Spherical Cordinates. shape: 1D
    :param Bx_data2: B_r of helio's data in Spherical Cordinates. shape: 3D
    :param By_data2: B_theta of helio's data in Spherical Cordinates. shape: 3D
    :param Bz_data2: B_phi of helio's data in Spherical Cordinates. shape: 3D
    :param r_i: r of target point in Spherical Coordinates. single value
    :param theta_i: theta of target point in Spherical Coordinates. single value
    :param phi_i: phi of target point in Spherical Coordinates. single value
    :param nv: the number of neighbor points. single value
    :param m: the number of polynomial functions (usually m<=nv). single value
    :param x_index_of_nearest_point: index of neareat point in r direction. single value
    :param y_index_of_nearest_point: index of neareat point in theta direction. single value
    :param z_index_of_nearest_point: index of neareat point in phi direction. single value
    :return: Brtp: magnetic field vector in in Spherical Coordinates. shape: 1D
    """
    x_data1, y_data1, z_data1 = \
        obtain_proper_length_xyz_data(x_data1_old, y_data1_old, z_data1_old, Bx_data1, By_data1, Bz_data1)
    x_data2, y_data2, z_data2 = \
        obtain_proper_length_xyz_data(x_data2_old, y_data2_old, z_data2_old, Bx_data2, By_data2, Bz_data2)

    # x_index_of_nearest_point = 1
    # y_index_of_nearest_point = 1
    # z_index_of_nearest_point = 1
    ri = get_xyz(r_i, theta_i, phi_i)
    xi = ri[0]
    yi = ri[1]
    zi = ri[2]
    Bi = np.zeros(3)
    cout = 0
    B_arr = np.zeros((3, nv))
    x = np.zeros(nv)
    y = np.zeros(nv)
    z = np.zeros(nv)
    if x_index_of_nearest_point == 0:
        for i in range(x_index_of_nearest_point - 1, x_index_of_nearest_point + 2):
            for j in range(y_index_of_nearest_point - 1, y_index_of_nearest_point + 2):
                for k in range(z_index_of_nearest_point - 1, z_index_of_nearest_point + 2):
                    # print(cout)
                    if i >= 0:
                        j, k = obtain_proper_nearest_index(j, k, y_data2, z_data2, Bx_data2, By_data2, Bz_data2)
                        # helio_i = np.argmin(np.abs(x_data2 - r_i))
                        # helio_j = np.argmin(np.abs(y_data2 - theta_i))
                        # helio_k = np.argmin(np.abs(z_data2 - phi_i))
                        Brtf = mag_from_rtp_to_xyz(Bx_data2[k, j, i], By_data2[k, j, i], Bz_data2[k, j, i], y_data2[j],
                                                   z_data2[k])
                        # B_arr[0, cout] = Bx_data2[i, j, k]
                        # B_arr[1, cout] = By_data2[i, j, k]
                        # B_arr[2, cout] = Bz_data2[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        x[cout] = x_data2[i] * np.sin(y_data2[j]) * np.cos(z_data2[k])
                        y[cout] = x_data2[i] * np.sin(y_data2[j]) * np.sin(z_data2[k])
                        z[cout] = x_data2[i] * np.cos(y_data2[j])
                        cout = cout + 1
                    else:
                        corona_i = np.argmin(np.abs(x_data1 - r_i))
                        corona_j = np.argmin(np.abs(y_data1 - theta_i))
                        corona_k = np.argmin(np.abs(z_data1 - phi_i))
                        corona_j, corona_k = obtain_proper_nearest_index(corona_j, corona_k, y_data1, z_data1)
                        Brtf = mag_from_rtp_to_xyz(Bx_data1[corona_k, corona_j, corona_i],
                                                   By_data1[corona_k, corona_j, corona_i],
                                                   Bz_data1[corona_k, corona_j, corona_i], y_data1[corona_j],
                                                   z_data1[corona_k])

                        # B_arr[0, cout] = Bx_data1[i, j, k]
                        # B_arr[1, cout] = By_data1[i, j, k]
                        # B_arr[2, cout] = Bz_data1[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        x[cout] = x_data1[corona_i] * np.sin(y_data1[corona_j]) * np.cos(z_data1[corona_k])
                        y[cout] = x_data1[corona_i] * np.sin(y_data1[corona_j]) * np.sin(z_data1[corona_k])
                        z[cout] = x_data1[corona_i] * np.cos(y_data1[corona_j])
                        cout = cout + 1

    if x_index_of_nearest_point == len(x_data1) - 1 and \
            y_index_of_nearest_point == np.argmin(np.abs(y_data1 - theta_i)) and \
            z_index_of_nearest_point == np.argmin(np.abs(z_data1 - phi_i)):
        for i in range(x_index_of_nearest_point - 1, x_index_of_nearest_point + 2):
            for j in range(y_index_of_nearest_point - 1, y_index_of_nearest_point + 2):
                for k in range(z_index_of_nearest_point - 1, z_index_of_nearest_point + 2):
                    # print(cout)
                    if i > (len(x_data1) - 1):
                        helio_i = np.argmin(np.abs(x_data2 - r_i))
                        helio_j = np.argmin(np.abs(y_data2 - theta_i))
                        helio_k = np.argmin(np.abs(z_data2 - phi_i))
                        helio_j, helio_k = obtain_proper_nearest_index(helio_j, helio_k, y_data2, z_data2)
                        Brtf = mag_from_rtp_to_xyz(Bx_data2[helio_k, helio_j, helio_i],
                                                   By_data2[helio_k, helio_j, helio_i],
                                                   Bz_data2[helio_k, helio_j, helio_i], y_data2[helio_j],
                                                   z_data2[helio_k])
                        # B_arr[0, cout] = Bx_data2[i, j, k]
                        # B_arr[1, cout] = By_data2[i, j, k]
                        # B_arr[2, cout] = Bz_data2[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        x[cout] = x_data2[helio_i] * np.sin(y_data2[helio_j]) * np.cos(z_data2[helio_k])
                        y[cout] = x_data2[helio_i] * np.sin(y_data2[helio_j]) * np.sin(z_data2[helio_k])
                        z[cout] = x_data2[helio_i] * np.cos(y_data2[helio_j])
                        cout = cout + 1
                    else:
                        j, k = obtain_proper_nearest_index(j, k, y_data1, z_data1)
                        Brtf = mag_from_rtp_to_xyz(Bx_data1[k, j, i], By_data1[k, j, i], Bz_data1[k, j, i], y_data1[j],
                                                   z_data1[k])
                        # B_arr[0, cout] = Bx_data1[i, j, k]
                        # B_arr[1, cout] = By_data1[i, j, k]
                        # B_arr[2, cout] = Bz_data1[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        x[cout] = x_data1[i] * np.sin(y_data1[j]) * np.cos(z_data1[k])
                        y[cout] = x_data1[i] * np.sin(y_data1[j]) * np.sin(z_data1[k])
                        z[cout] = x_data1[i] * np.cos(y_data1[j])
                        cout = cout + 1
    else:
        if r_i >= x_data2[0]:
            for i in range(x_index_of_nearest_point - 1, x_index_of_nearest_point + 2):
                for j in range(y_index_of_nearest_point - 1, y_index_of_nearest_point + 2):
                    for k in range(z_index_of_nearest_point - 1, z_index_of_nearest_point + 2):
                        # print(cout)
                        if x_index_of_nearest_point + 1 >= len(x_data2):
                            print('Current point is beyond the external boundary of helio!')
                            return 0
                        j, k = obtain_proper_nearest_index(j, k, y_data2, z_data2)
                        Brtf = mag_from_rtp_to_xyz(Bx_data2[k, j, i], By_data2[k, j, i], Bz_data2[k, j, i], y_data2[j],
                                                   z_data2[k])
                        # B_arr[0, cout] = Bx_data2[i, j, k]
                        # B_arr[1, cout] = By_data2[i, j, k]
                        # B_arr[2, cout] = Bz_data2[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        # print(x_data[i])
                        # x[cout] = x_data[i]
                        # y[cout] = y_data[j]
                        # z[cout] = z_data[k]
                        x[cout] = x_data2[i] * np.sin(y_data2[j]) * np.cos(z_data2[k])
                        y[cout] = x_data2[i] * np.sin(y_data2[j]) * np.sin(z_data2[k])
                        z[cout] = x_data2[i] * np.cos(y_data2[j])
                        # print(x[cout])
                        cout = cout + 1
        if r_i < x_data1[-1]:
            # print(x_index_of_nearest_point)
            # print(np.shape(Bx_data1))
            for i in range(x_index_of_nearest_point - 1, x_index_of_nearest_point + 2):
                for j in range(y_index_of_nearest_point - 1, y_index_of_nearest_point + 2):
                    for k in range(z_index_of_nearest_point - 1, z_index_of_nearest_point + 2):
                        # print(cout)
                        j, k = obtain_proper_nearest_index(j, k, y_data1, z_data1)
                        Brtf = mag_from_rtp_to_xyz(Bx_data1[k, j, i], By_data1[k, j, i], Bz_data1[k, j, i], y_data1[j],
                                                   z_data1[k])
                        # B_arr[0, cout] = Bx_data1[i, j, k]
                        # B_arr[1, cout] = By_data1[i, j, k]
                        # B_arr[2, cout] = Bz_data1[i, j, k]
                        B_arr[0, cout] = Brtf[0]
                        B_arr[1, cout] = Brtf[1]
                        B_arr[2, cout] = Brtf[2]
                        # print(x_data[i])
                        # x[cout] = x_data[i]
                        # y[cout] = y_data[j]
                        # z[cout] = z_data[k]
                        x[cout] = x_data1[i] * np.sin(y_data1[j]) * np.cos(z_data1[k])
                        y[cout] = x_data1[i] * np.sin(y_data1[j]) * np.sin(z_data1[k])
                        z[cout] = x_data1[i] * np.cos(y_data1[j])
                        # print(x[cout])
                        cout = cout + 1
                # end
            # end
    # end

    # 以下不需要改动
    r = np.zeros((nv))
    for j in range(nv):
        r[j] = np.sqrt((xi - x[j]) ** 2 + (yi - y[j]) ** 2 + (zi - z[j]) ** 2)
    r_max = np.nanmax(r)
    r_support = 2 * r_max

    PHI = np.zeros((3, 3, nv, nv))
    for j in range(nv):
        for k in range(nv):
            psi = RBF_function_partial1(r_support, x[j], y[j], z[j],
                                        x[k], y[k], z[k])
            for j1 in [0, 1, 2]:
                for k1 in [0, 1, 2]:
                    PHI[j1, k1, j, k] = psi[j1, k1]
    vector = np.zeros(3)
    P = np.zeros((3 * nv, m))
    for j in range(nv):
        for k in range(m):
            psi = polynomial_magnetic(x[j], y[j], z[j], k)
            for j1 in [0, 1, 2]:
                vector[j1] = psi[j1]
            j1 = (j) * 3
            # print(vector)
            # input()
            # P[j1+1:j1+4,k] = vector[:]
            P[j1 + 0, k] = vector[0]
            P[j1 + 1, k] = vector[1]
            P[j1 + 2, k] = vector[2]

    PT = np.transpose(P)
    A = np.zeros((3 * nv + m, 3 * nv + m))
    c = np.zeros(3 * nv + m)
    for j in range(0, 3 * nv - 2, 3):
        for k in range(0, 3 * nv - 2, 3):
            for j1 in [j, j + 1, j + 2]:
                for k1 in [k, k + 1, k + 2]:
                    # print(k1)
                    A[j1, k1] = PHI[j1 - j, k1 - k, int(j / 3), int(k / 3)]  # 不需要加1

    for j in range(3 * nv):
        for k in range(3 * nv, 3 * nv + m):
            A[j, k] = P[j, k - 3 * nv]
    for j in range(3 * nv, 3 * nv + m):
        for k in range(3 * nv):
            A[j, k] = PT[j - 3 * nv, k]
    for j in range(0, 3 * nv - 2, 3):
        for j1 in [j, j + 1, j + 2]:
            c[j1] = B_arr[j1 - j, int(j / 3)]  # 不需要加1
    w = svd_solver(A, c)
    for j in range(nv):
        psi = RBF_function_partial1(r_support, xi, yi, zi,
                                    x[j], y[j], z[j])
        for k in range(3):
            Bi[0] = Bi[0] + psi[0, k] * w[(j) * 3 + k]
            Bi[1] = Bi[1] + psi[1, k] * w[(j) * 3 + k]
            Bi[2] = Bi[2] + psi[2, k] * w[(j) * 3 + k]
    for j in range(m):
        psi = polynomial_magnetic(xi, yi, zi, j)
        Bi[0] = Bi[0] + psi[0] * w[3 * nv + j]
        Bi[1] = Bi[1] + psi[1] * w[3 * nv + j]
        Bi[2] = Bi[2] + psi[2] * w[3 * nv + j]

    Bi_in_rtf = mag_from_xyz_to_rtf(Bi[0], Bi[1], Bi[2], theta_i, phi_i)
    return Bi_in_rtf
