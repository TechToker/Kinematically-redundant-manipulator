import numpy as np
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


link_length = [1, 1, 1, 1]


def FK(q, links):
    T = np.linalg.multi_dot([Rz(q[0]),
                             Tx(links[0]),
                             Rz(q[1]),
                             Tx(links[1]),
                             Rz(q[2]),
                             Tx(links[2]),
                             Rz(q[3]),
                             Tx(links[3])])

    return T


def GetJacobianColumn(dT, T_inv):
    dT = np.linalg.multi_dot([dT, T_inv])
    return np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])


def JacobianVirtual(q, links):
    T = FK(q, links)
    T[0:3, 3] = 0
    T_inv = np.transpose(T)

    dT = np.linalg.multi_dot([dRz(q[0]),
                              Tx(links[0]),
                              Rz(q[1]),
                              Tx(links[1]),
                              Rz(q[2]),
                              Tx(links[2]),
                              Rz(q[3]),
                              Tx(links[3])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J1 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Rz(q[0]),
                              Tx(links[0]),
                              dRz(q[1]),
                              Tx(links[1]),
                              Rz(q[2]),
                              Tx(links[2]),
                              Rz(q[3]),
                              Tx(links[3])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J2 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Rz(q[0]),
                              Tx(links[0]),
                              Rz(q[1]),
                              Tx(links[1]),
                              dRz(q[2]),
                              Tx(links[2]),
                              Rz(q[3]),
                              Tx(links[3])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J3 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Rz(q[0]),
                              Tx(links[0]),
                              Rz(q[1]),
                              Tx(links[1]),
                              Rz(q[2]),
                              Tx(links[2]),
                              dRz(q[3]),
                              Tx(links[3])])

    dT = np.linalg.multi_dot([dT, T_inv])
    J4 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    J = np.hstack([J1, J2, J3, J4])
    return J


def PlotFK(q, links):
    pos0 = [0, 0]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tx(links[0])])

    pos1 = T[0:3, 3]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tx(links[0]),
                             Rz(q[1]),
                             Tx(links[1])])

    pos2 = T[0:3, 3]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tx(links[0]),
                             Rz(q[1]),
                             Tx(links[1]),
                             Rz(q[2]),
                             Tx(links[2])])

    pos3 = T[0:3, 3]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tx(links[0]),
                             Rz(q[1]),
                             Tx(links[1]),
                             Rz(q[2]),
                             Tx(links[2]),
                             Rz(q[3]),
                             Tx(links[3])])

    pos4 = T[0:3, 3]
    print(f"End-effector pos: {pos4}")

    x = [pos0[0], pos1[0], pos2[0], pos3[0], pos4[0]]
    y = [pos0[1], pos1[1], pos2[1], pos3[1], pos4[1]]

    plt.xlim(-3, 3)
    plt.ylim(-3.25, 3.25)
    plt.plot(x, y)
    plt.scatter(x, y)


# Print first fk
r_global = np.array([1, 3, 0, 0, 0, 0])
PlotFK([0, np.pi / 2, 0, 0], link_length)


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
        J_wgh_1 = np.linalg.multi_dot([np.linalg.inv(weighs), np.transpose(jac)])
        J_wgh_2 = np.linalg.multi_dot([jac, np.linalg.inv(weighs), np.transpose(jac)])

        J_wgh = np.linalg.multi_dot([J_wgh_1, np.linalg.pinv(J_wgh_2)])

        delta_q = np.dot(J_wgh, d_error)

        q_current = q_current + delta_q
        i += 1

    return q_current


q_start = np.array([np.pi / 6, np.pi / 2, -np.pi / 6, 0])
weighs_pseudo_inv = np.diag([0.01, 1000, 1000, 1000])

q_final = WeightedPseudoInv(q_start, weighs_pseudo_inv)

# Print second fk
PlotFK(q_final, link_length)
plt.show()
