import numpy as np
from math import *
import matplotlib.pyplot as plt


def Rx(q):
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q), np.cos(q), 0],
                  [0, 0, 0, 1]])
    return T


def Ry(q):
    T = np.array([[np.cos(q), 0, np.sin(q), 0],
                  [0, 1, 0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 1]])
    return T


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q), np.cos(q), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    return T


def dRx(q):
    T = np.array([[0, 0, 0, 0],
                  [0, -np.sin(q), -np.cos(q), 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, 0, 0, 0]])
    return T


def dRy(q):
    T = np.array([[-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 0],
                  [-np.cos(q), 0, -np.sin(q), 0],
                  [0, 0, 0, 0]])
    return T


def dRz(q):
    T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                  [np.cos(q), -np.sin(q), 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def dTx(x):
    T = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def dTy(y):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def dTz(z):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    return T


link_length = [0.140, 0.200, 0.200, 0.200, 0.200, 0.200, 0.126]


def FK(q, links):
    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    return T


def GetJacobianColumn(dT, T_inv):
    dT = np.linalg.multi_dot([dT, T_inv])
    return np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])


def JacobianVirtual(q, links):
    T = FK(q, links)
    T[0:3, 3] = 0
    T_inv = np.transpose(T)

    dT = np.linalg.multi_dot([Tz(links[0]),
                             dRz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J1 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             dRy(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])


    dT = np.linalg.multi_dot([dT, T_inv])
    J2 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             dRz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])


    dT = np.linalg.multi_dot([dT, T_inv])
    J3 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             dRy(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J4 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             dRz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J5 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             dRy(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J6 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             dRz(q[6])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J7 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    J = np.hstack([J1, J2, J3, J4, J5, J6, J7])
    return J


ax = plt.axes(projection='3d')


def PlotFK(q, links, color='b'):
    pos0 = [0, 0, 0]

    T = Tz(links[0])

    pos1 = T[0:3, 3]

    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1])])

    pos2 = T[0:3, 3]

    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2])])
    pos3 = T[0:3, 3]

    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3])])

    pos4 = T[0:3, 3]

    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4])])

    pos5 = T[0:3, 3]

    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5])])

    pos6 = T[0:3, 3]


    T = np.linalg.multi_dot([Tz(links[0]),
                             Rz(q[0]),
                             Tz(links[1]),
                             Ry(q[1]),
                             Tz(links[2]),
                             Rz(q[2]),
                             Tz(links[3]),
                             Ry(q[3]),
                             Tz(links[4]),
                             Rz(q[4]),
                             Tz(links[5]),
                             Ry(q[5]),
                             Tz(links[6]),
                             Rz(q[6])])

    pos7 = T[0:3, 3]

    print(f"End-effector pos: {pos7}")

    x = [pos0[0], pos1[0], pos2[0], pos3[0], pos4[0], pos5[0], pos6[0], pos7[0]]
    y = [pos0[1], pos1[1], pos2[1], pos3[1], pos4[1], pos5[1], pos6[1], pos7[1]]
    z = [pos0[2], pos1[2], pos2[2], pos3[2], pos4[2], pos5[2], pos6[2], pos7[2]]

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)

    ax.plot3D(x, y, z, linewidth=0.5, c=color)
    #ax.scatter3D(x, y, z, s=25, c='lightcoral')


# Print first fk

def WeightedPseudoInv(q_current, weighs):
    i = 0
    error = [10, 10, 10, 10, 10, 10]

    while abs(sum(error[0:3])) > 0.01 or i < 2:
        r_current = FK(q_current, link_length)
        r_current = np.hstack([r_current[0:3, 3], [0, 0, 0]])

        error = r_global - r_current
        print(f"[{i}] Error sum: {round(sum(error[0:3]), 4)}")
        d_error = error / 100

        jac = JacobianVirtual(q_current, link_length)
        print(jac.shape)

        J_wgh_1 = np.linalg.multi_dot([np.linalg.pinv(weighs), np.transpose(jac)])
        J_wgh_2 = np.linalg.multi_dot([jac, np.linalg.pinv(weighs), np.transpose(jac)])

        J_wgh = np.linalg.multi_dot([J_wgh_1, np.linalg.pinv(J_wgh_2)])

        delta_q = np.dot(J_wgh, d_error)
        print(delta_q.shape)

        q_current = q_current + delta_q
        i += 1

    return q_current


def TaskPrior(q_current):
    i = 0
    error = [10, 10, 10, 10, 10, 10]

    #r_global_fix = r_global
    r_global_fix = np.hstack([r_global, [r_global[0], r_global[1], r_global[2], np.pi/2, 0, 0]])

    while abs(sum(error[:])) > 0.01 or i < 2:
        r_current = FK(q_current, link_length)
        r_current = np.hstack([r_current[0:3, 3], [0, 0, 0], r_current[0:3, 3], [np.pi/2, 0, 0]])

        error = r_global_fix - r_current
        print(f"[{i}] Error sum: {round(sum(error[:]), 4)}")
        d_error = error / 100

        jac1 = JacobianVirtual(q_current, link_length)

        q_need = q_current
        jac2 = JacobianVirtual(q_need, link_length)

        J_full = np.vstack([jac1, jac2])
        J_pinv = np.linalg.pinv(J_full)

        delta_q = np.dot(J_pinv, d_error)

        q_current = q_current + delta_q
        i += 1

    return q_current


def DLS(q_current, links_length):
    i = 0
    error = [10, 10, 10, 10, 10, 10]

    nu = 0.1
    Im = np.ones(6)

    #for i in range (0, 1000):
    while abs(sum(error[0:3])) > 0.01 or i < 2:
        r_current = FK(q_current, link_length)
        r_current = np.hstack([r_current[0:3, 3], [0, 0, 0]])

        error = r_global - r_current
        print(f"[{i}] Error sum: {round(sum(error[0:3]), 4)}")
        d_error = error / 100

        J = JacobianVirtual(q_current, links_length)
        J_pinv = np.dot(np.transpose(J), np.linalg.pinv(np.dot(J, np.transpose(J))+nu**2 * Im))

        delta_q = np.dot(J_pinv, d_error)

        q_current = q_current + delta_q
        i += 1

    return q_current


def GetH(jacob):
    H = np.sqrt(np.linalg.det(np.linalg.multi_dot([jacob, np.transpose(jacob)])))
    return H

def NullSpace(q_current, links_length):
    PlotFK(q_current, links_length, color="black")

    i = 0
    error = [10, 10, 10, 10, 10, 10]

    nu = 0.1
    Im = np.ones(6)

    #for i in range (0, 50):
    while abs(sum(error[0:3])) > 0.01 or i < 2:
        r_current = FK(q_current, link_length)
        r_current = np.hstack([r_current[0:3, 3], [0, 0, 0]])

        error = r_global - r_current
        print(f"[{i}] Error sum: {sum(error[0:3])}")
        d_error = error / 100

        jac = JacobianVirtual(q_current, links_length)
        jac_pinv = np.linalg.pinv(jac)

        H_init = GetH(jac)

        delta_q = 0.001

        q_dot_zero = None

        for j in range(len(jac[0])):
            v_cur = q_current.copy()
            v_cur[0] += delta_q

            jac_cur = JacobianVirtual(v_cur, links_length)
            H_cur = GetH(jac_cur)

            q_dot_zero_cur = (H_cur - H_init) / delta_q

            if q_dot_zero is None:
                q_dot_zero = q_dot_zero_cur
            else:
                q_dot_zero = np.hstack([q_dot_zero, q_dot_zero_cur])

        term01 = np.dot(jac_pinv, jac)

        Im = np.ones((7, 7))
        term0 = Im - term01

        delta_q = np.dot(jac_pinv, d_error) + np.dot(term0, q_dot_zero)

        q_current = q_current + delta_q
        i += 1

        if i == 1:
            PlotFK(q_current, links_length, color="pink")

        if i % 50 == 0:
            PlotFK(q_current, links_length, color="green")

    return q_current

trajectory_points = 5

# trajectory_x = []
# trajectory_x.extend(np.linspace(-0.25, 0.25, num=trajectory_points))
# trajectory_x.extend([0.25] * trajectory_points)
# trajectory_x.extend(np.linspace(0.25, -0.25, num=trajectory_points))
# trajectory_x.extend([-0.25] * trajectory_points)
#
# trajectory_y = []
# trajectory_y.extend([0.65] * trajectory_points)
# trajectory_y.extend([0.65] * trajectory_points)
# trajectory_y.extend([0.65] * trajectory_points)
# trajectory_y.extend([0.65] * trajectory_points)
#
# trajectory_z = []
# trajectory_z.extend([0.1] * trajectory_points)
# trajectory_z.extend(np.linspace(0.1, 0.65, num=trajectory_points))
# trajectory_z.extend([0.65] * trajectory_points)
# trajectory_z.extend(np.linspace(0.65, 0.1, num=trajectory_points))


trajectory_x = []
trajectory_x.extend(np.linspace(-0.25, 0.25, num=trajectory_points))
trajectory_x.extend([0.25] * trajectory_points)
trajectory_x.extend(np.linspace(0.25, -0.25, num=trajectory_points))
trajectory_x.extend([-0.25] * trajectory_points)

trajectory_z = []
trajectory_z.extend([1.1] * trajectory_points)
trajectory_z.extend([1.1] * trajectory_points)
trajectory_z.extend([1.1] * trajectory_points)
trajectory_z.extend([1.1] * trajectory_points)

trajectory_y = []
trajectory_y.extend([0.25] * trajectory_points)
trajectory_y.extend(np.linspace(-0.25, 0.25, num=trajectory_points))
trajectory_y.extend([-0.25] * trajectory_points)
trajectory_y.extend(np.linspace(0.25, -0.25, num=trajectory_points))

trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)
trajectory_z = np.array(trajectory_z)

ax.scatter3D(trajectory_x, trajectory_y, trajectory_z, s=25, c='blue')

weighs_pseudo_inv = np.diag([1, 1, 1, 1, 1, 1, 1])

q_start = np.array([-0.04208818, -2.16033204,  1.88463568,  1.45515167,  2.47635915, -1.78764181, -1.01590906])

for i in range(len(trajectory_x)):
    print(trajectory_x.shape)
    print(trajectory_x[i])

    r_global = np.array([trajectory_x[i], trajectory_y[i], trajectory_z[i], 0, 0, 0])
    q_final = TaskPrior(q_start)

    PlotFK(q_final, link_length, 'r')

#PlotFK([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2], link_length, color="blue")



#r_global = np.array([-0.4, 0.4, 0.466, 0, 0, 0])


#weighs_pseudo_inv = np.diag([1, 1, 1, 1, 1, 1, 1])

#q_final = WeightedPseudoInv(q_start, weighs_pseudo_inv)

#q_final = DLS(q_start, link_length)

#q_final = NullSpace(q_start, link_length)

# Print second fk
#PlotFK(q_final, link_length, 'r')
plt.show()

