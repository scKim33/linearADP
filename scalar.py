import numpy as np
from scipy.io import loadmat

class model_antenna:
    """
    Satellite tracking antenna
    theta: antenna angle from the origin
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque from the drive motor
    :param J: a moment of inertia of antenna and driving parts
    :param B: a damping of the system
    :return: x_dot
    """

    def __init__(self, x_ref):
        self.J = 600000  # [kg * m^2]
        self.B = 20000  # [N * m * sec]
        self.A = np.array([[0, 1],
                           [0, -self.B / self.J]])
        self.B = np.array([[0],
                           [1 / self.J]])
        self.x_ref = x_ref

    def dynamics(self, x, t, u):
        # state - space specification
        x_dot = np.array([x[1], -self.B / self.J * x[1] + u / self.J])
        return x_dot


class model_pendulum:
    """
    Pendulum system with torque
    theta: Angle of the pendulum from the gravity direction
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque about the pivot point
    :param g: Acceleration of gravity
    :param l: Pendulum length
    :return: x_dot
    """
    def __init__(self, x_ref):
        self.g = 9.81  # [m / s^2]
        self.l = 20  # [m]
        self.m = 0.1  # [kg]
        self.A = np.array([[0, 1],
                           [-self.g / self.l, 0]])
        self.B = np.array([[0],
                           [1 / (self.m * np.power(self.l, 2))]])
        # augmented system with e_I dynamics
        self.Aa = np.array([[0, 1, 0],
                            [-self.g / self.l, 0, 0],
                            [1, 0, 0]])  # performance output
        self.Ba = np.array([[0],
                            [1 / (self.m * np.power(self.l, 2))],
                            [0]])
        self.x_ref = x_ref

    def dynamics(self, x, t, u):
        # state - space specification
        x_dot = np.array([x[1],
                          -self.g / self.l * x[0] + u / (self.m * np.power(self.l, 2))])
        return x_dot

    def aug_dynamics(self, x, t, u):
        # state - space specification
        x_dot = np.array([x[1],
                          -self.g / self.l * x[0] + u / (self.m * np.power(self.l, 2)),
                          x[0] - self.x_ref])
        return x_dot


class model_f18_lat:
    def __init__(self, x_ref):
        self.mat = loadmat('dat/f18_lin_data.mat')
        self.A = self.mat['Alon']
        self.B = self.mat['Blon']
        self.C = np.array([[1, 0, 0, 0]])   # x_reference for velocity
        self.x_trim = self.mat['x_trim_lon']
        self.x_ref = x_ref
        self.Aa = np.block([[self.A, np.zeros((np.shape(self.A)[0], np.shape(self.C)[0]))],
                            [self.C, np.zeros((np.shape(self.C)[0], np.shape(self.C)[0]))]])
        self.Ba = np.block([[self.B],
                            [np.zeros((np.shape(self.C)[0], np.shape(self.B)[1]))]])

    def dynamics(self, x, t, u):
        return (np.dot(self.A, x) + np.dot(self.B, u)).squeeze()

    def aug_dynamics(self, x, t, u):
        aug_x = np.block([[np.reshape(x, (len(x), 1))]])    # x is already augmented at test_scalar.py, so is not required to add zeros
        aug_x_ref = np.block([[np.zeros((4, 1))],
                              [self.x_ref]])
        return np.dot(self.Aa, aug_x).squeeze() + np.dot(self.Ba, u) - aug_x_ref.squeeze()
