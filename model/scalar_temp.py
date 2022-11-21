import numpy as np


class antenna:
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


class pendulum:
    """
    Pendulum system with torque
    theta: Angle of the pendulum from the gravity direction
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque about the pivot point
    :param g: Acceleration of gravity
    :param l: Pendulum length
    :return: x_dot
    """

    def __init__(
            self,
            x0=None,
            x_ref=None,
            C=np.diag([1, 1]),
            Q=np.diag([1e5, 1]),
            R=np.diag([1]),
            Qa=np.diag([100, 10, 100, 100]),
            Ra=np.diag([1])
    ):
        # initial x_ref setting from trim condition
        # x_ref default value : [some random value, 0]
        if x_ref is not None:
            self.x_ref = x_ref
        else:
            self.x_ref = np.array([np.random.normal(0, np.deg2rad(2)),
                                   0])
        noise = np.array([np.random.normal(0, np.deg2rad(0.5)),
                          np.random.normal(0, np.deg2rad(0.5))])
        # x0 default value : noise
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = noise
        self.g = 9.81  # [m / s^2]
        self.l = 20  # [m]
        self.m = 0.1  # [kg]
        self.A = np.array([[0, 1],
                           [-self.g / self.l, 0]])
        self.B = np.array([[0],
                           [1 / (self.m * np.power(self.l, 2))]])
        self.C = C
        self.Q = Q
        self.R = R
        # augmented system with e_I dynamics
        self.Aa = np.block([[self.A, np.zeros((np.shape(self.A)[0], np.shape(self.C)[0]))],
                            [self.C, np.zeros((np.shape(self.C)[0], np.shape(self.C)[0]))]])
        self.Ba = np.block([[self.B],
                            [np.zeros((np.shape(self.C)[0], np.shape(self.B)[1]))]])
        self.Qa = Qa
        self.Ra = Ra

    def dynamics(self, x, t, u):
        # state - space specification
        x_dot = np.array([x[1],
                          -self.g / self.l * x[0] + u / (self.m * np.power(self.l, 2))])
        return x_dot

    def aug_dynamics(self, x, t, u):
        aug_x = np.block(
            [[np.reshape(x, (len(x), 1))]])  # x is already augmented at test_scalar.py, so is not required to add zeros
        aug_x_ref = np.block([[np.zeros((np.shape(self.A)[0], 1))],
                              [np.reshape(self.C @ self.x_ref,
                                          (np.shape(self.C)[0], 1))]])  # x_ref shape : (# row of C,) -> (# row of C, 1)
        return np.dot(self.Aa, aug_x).squeeze() + np.dot(self.Ba, u).squeeze() - aug_x_ref.squeeze()

class dc_motor:
    def __init__(
            self,
            x0=None,
            x_ref=None,
            C=np.diag([1, 1]),
            Q=np.diag([1, 1]),
            R=np.diag([1]),
    ):
        self.name = 'dc_motor'
        # initial x_ref setting from trim condition
        # x_ref default value : [some random value, 0]
        if x_ref is not None:
            self.x_ref = x_ref
        else:
            self.x_ref = np.zeros((2, 1))
        noise = 3 * np.random.randn(2, 1)
        # x0 default value : noise
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = noise
        self.A = np.array([[-10, 1],
                           [-0.002, -2]])
        self.B = np.array([[0],
                           [2]])
        self.C = C
        self.Q = Q
        self.R = R

    def dynamics(self, x, t, u):
        return (np.dot(self.A, x) + np.dot(self.B, u).squeeze()).squeeze()