"""
Description: NAO Analytic Motion Retargeting Class
Reference: "A Simple and Fast Geometric Kinematic Solution for Imitation of Human Arms by a NAO Humanoid Robot"
Author: Tae-woo Kim
Contact: twkim0812@gmail.com
"""

from functools import reduce
from numpy.linalg import norm
from CommonObject import *
PI = np.pi


def iData(data, index):
    idx = index * 3
    return np.array(data[idx:idx+3])


def Rx(deg):
    return np.array([[1, 0, 0], [0, np.cos(d2r(deg)), -np.sin(d2r(deg))], [0, np.sin(d2r(deg)), np.cos(d2r(deg))]])


def Ry(deg):
    return np.array([[np.cos(d2r(deg)), 0, np.sin(d2r(deg))], [0, 1, 0], [-np.sin(d2r(deg)), 0, np.cos(d2r(deg))]])


def Rz(deg):
    return np.array([[np.cos(d2r(deg)), -np.sin(d2r(deg)), 0], [np.sin(d2r(deg)), np.cos(d2r(deg)), 0], [0, 0, 1]])


class NAO_AMR:
    def __init__(self):
        # NAO Arm length (meter)
        self.L1 = 0.015     # shoulder shift link
        self.L2 = 0.105     # the upper limb
        self.L3 = 0.055     # the lower limb

        # DH-parameter setup left arm [theta(z), alpha(x), d(z), a(x)]
        self.lDHparam = np.zeros((5, 4))
        # self.lDHparam[0, :] = np.array([0,              -d2r(90),   0.1,    0])   # torso to neck
        # self.lDHparam[1, :] = np.array([0,              0,          0.0978, 0])   # neck to left shoulder

        self.lDHparam[0, :] = np.array([0,              d2r(90),    0,          0])
        self.lDHparam[1, :] = np.array([0 + d2r(90),    d2r(90),    0,          self.L1])
        self.lDHparam[2, :] = np.array([0 + d2r(180),   d2r(90),    self.L2,    0])
        self.lDHparam[3, :] = np.array([0 + d2r(180),   d2r(90),    0,          0])
        self.lDHparam[4, :] = np.array([0,              0,          self.L3,    0])

        # DH-parameter setup of right arm [theta(z), alpha(x), d(z), a(x)]
        self.rDHparam = np.zeros((5, 4))
        # self.rDHparam[0, :] = np.array([0,              -d2r(90),   0.1,    0])  # torso to neck
        # self.rDHparam[1, :] = np.array([0,              0,         -0.0978, 0])  # neck to left shoulder

        self.rDHparam[0, :] = np.array([0,              d2r(90),    0,          0])
        self.rDHparam[1, :] = np.array([0 - d2r(90),   -d2r(90),    0,          self.L1])
        self.rDHparam[2, :] = np.array([0,              d2r(90),    self.L2,    0])
        self.rDHparam[3, :] = np.array([0 + d2r(180),   d2r(90),    0,          0])
        self.rDHparam[4, :] = np.array([0,              0,          self.L3,    0])
        self.DOF = self.lDHparam.shape[0]

        # Left arm Transformation Matrix
        self.lPT = np.zeros((self.DOF + 1, 4, 4))
        self.lA = np.zeros((self.DOF + 1, 4, 4))

        # Right arm Transformation Matrix
        self.rPT = np.zeros((self.DOF + 1, 4, 4))
        self.rA = np.zeros((self.DOF + 1, 4, 4))

        self.init_tr_matrix()

    def init_tr_matrix(self):
        # Left Arm transformation matrix initialization
        for i in range(0, self.lDHparam.shape[0]):
            theta = self.lDHparam[i, 0]
            alpha = self.lDHparam[i, 1]
            d = self.lDHparam[i, 2]
            a = self.lDHparam[i, 3]
            lTrz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
            lRotz = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            lTrx = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            lRotx = np.array([[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])
            self.lPT[i+1] = reduce(np.matmul, [lTrz, lRotz, lTrx, lRotx])

            if i == 0:
                self.lA[i] = self.lPT[i] = np.identity(4)
            self.lA[i+1] = reduce(np.matmul, [self.lA[i], self.lPT[i+1]])
        self.lA[-1] = reduce(np.matmul, [self.lA[-2], self.lPT[-1]])

        # Right Arm transformation matrix initialization
        for i in range(0, self.rDHparam.shape[0]):
            theta = self.rDHparam[i, 0]
            alpha = self.rDHparam[i, 1]
            d = self.rDHparam[i, 2]
            a = self.rDHparam[i, 3]
            rTrz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
            rRotz = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            rTrx = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            rRotx = np.array([[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])
            self.rPT[i + 1] = reduce(np.matmul, [rTrz, rRotz, rTrx, rRotx])

            if i == 0:
                self.rA[i] = self.rPT[i] = np.identity(4)
            self.rA[i + 1] = reduce(np.matmul, [self.rA[i], self.rPT[i + 1]])
        self.rA[-1] = reduce(np.matmul, [self.rA[-2], self.rPT[-1]])

    def disp_mat(self):
        print(" ===== Display Left Arm PT matrices =====")
        for m in self.lPT:
            print(m)
        print("\n")
        print(" ===== Display Left Arm A matrices =====")
        for m in self.lA:
            print(m)

        print(" ===== Display Right Arm PT matrices =====")
        for m in self.rPT:
            print(m)
        print("\n")
        print(" ===== Display Right Arm A matrices =====")
        for m in self.rA:
            print(m)

    def left_arm_solve(self, data):
        # Human Upper Left arm vector
        hUL = iData(data, 5) - iData(data, 4)
        hUL = np.matmul(np.transpose(Rx(-90)), hUL)

        # Robot Upper Left arm vector
        rUL = self.lA[3, 0:3, 3]
        rUL = (norm(rUL) / norm(hUL)) * hUL

        # angles
        q2 = np.arcsin(rUL[2] / np.sqrt(self.L1 ** 2 + self.L2 ** 2)) + np.arctan(-self.L1 / self.L2)
        q1 = np.arctan2(rUL[1] / (self.L2 * np.cos(q2) - self.L1 * np.sin(q2)), rUL[0] / (self.L2 * np.cos(q2) - self.L1 * np.sin(q2)))

        # Human Lower Left arm vector
        Q1 = np.array([[np.cos(q1), 0, np.sin(q1)], [np.sin(q1), 0, -np.cos(q1)], [0, 1, 0]])
        Q2 = np.array([[-np.sin(q2), 0, np.cos(q2)], [np.cos(q2), 0, np.sin(q2)], [0, 1, 0]])
        hLL = iData(data, 6) - iData(data, 5)
        hLL = reduce(np.matmul, [Q2.transpose(), Q1.transpose(), Rx(-90).transpose(), hLL])

        rLL = reduce(np.matmul, [self.lPT[3, 0:3, 0:3], self.lPT[4, 0:3, 0:3], self.lPT[5, 0:3, 3]])
        rLL = (norm(rLL) / norm(hLL)) * hLL

        q4 = np.arctan2(-np.sqrt(rLL[0] ** 2 + rLL[1] ** 2) / self.L3, rLL[2] / self.L3)
        q3 = np.arctan2(-rLL[1] / np.sqrt(rLL[0] ** 2 + rLL[1] ** 2), -rLL[0] / np.sqrt(rLL[0] ** 2 + rLL[1] ** 2))

        return [q1, q2, q3, q4]

    def right_arm_solve(self, data):
        # Human Upper Right arm vector
        hUR = iData(data, 9) - iData(data, 8)
        hUR = np.matmul(np.transpose(Rx(-90)), hUR)

        # Robot Upper Right arm vector
        rUR = self.rA[3, 0:3, 3]
        rUR = (norm(rUR) / norm(hUR)) * hUR

        # angles
        q2 = np.arcsin(rUR[2] / np.sqrt(self.L1 ** 2 + self.L2 ** 2)) + np.arctan(self.L1 / self.L2)
        q1 = np.arctan2(rUR[1] / (self.L2 * np.cos(q2) + self.L1 * np.sin(q2)), rUR[0] / (self.L2 * np.cos(q2) + self.L1 * np.sin(q2)))

        # Human Lower Left arm vector
        Q1 = np.array([[np.cos(q1), 0, np.sin(q1)], [np.sin(q1), 0, -np.cos(q1)], [0, 1, 0]])
        Q2 = np.array([[np.sin(q2), 0, np.cos(q2)], [-np.cos(q2), 0, np.sin(q2)], [0, -1, 0]])

        hLR = iData(data, 10) - iData(data, 9)
        hLR = reduce(np.matmul, [Q2.transpose(), Q1.transpose(), Rx(-90).transpose(), hLR])

        rLR = reduce(np.matmul, [self.rPT[3, 0:3, 0:3], self.rPT[4, 0:3, 0:3], self.rPT[5, 0:3, 3]])
        rLR = (norm(rLR) / norm(hLR)) * hLR

        q4 = np.arctan2(np.sqrt(rLR[0] ** 2 + rLR[1] ** 2) / self.L3, rLR[2] / self.L3)
        q3 = np.arctan2(-rLR[1] / np.sqrt(rLR[0] ** 2 + rLR[1] ** 2), -rLR[0] / np.sqrt(rLR[0] ** 2 + rLR[1] ** 2))

        return [q1, q2, q3, q4]
