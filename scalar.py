import numpy as np


def model_antenna(x, t, u):
    """
    Satellite tracking antenna
    theta: antenna angle from the origin
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque from the drive motor, given by PID controller
    :param J: a moment of inertia of antenna and driving parts
    :param B: a damping of the system
    :return: x_dot
    """
    J = 600000  # [kg * m^2]
    B = 20000  # [N * m * sec]

    # state - space specification
    x_dot = np.array([x[1], -B / J * x[1] + u / J])

    return x_dot


def model_pendulum(x, t, u):
    """
    Pendulum system with torque
    theta: Angle of the pendulum from the gravity direction
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque about the pivot point, given by PID controller
    :param g: Acceleration of gravity
    :param l: Pendulum length
    :return: x_dot
    """
    g = 9.81  # [m / s^2]
    l = 20  # [m]
    m = 0.1  # [kg]

    # state - space specification
    x_dot = np.array([x[1], -g / l * x[0] + u / (m * np.power(l, 2))])

    return x_dot
