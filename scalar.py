import time
import matplotlib.pyplot as plt
from simple_pid import PID
import numpy as np
from scipy.integrate import odeint


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


def sim(t_end, t_step, model, x0, controller):
    """
    Linear model simulation
    :param model: State-space form linear model
    :param args: Tuple; coefficient used at the model, careful to the input sequence
    :param controller: Two kinds of controller ; "PID" or "LQR"
    :param x0: Initial condition of the system
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :return: System variable x (vector form)
    """

    x = x0
    u = 0
    x_hist = x
    u_hist = u
    for t in tspan:
        ts = [t, t + t_step]
        if controller == "PID":
            # simulation condition -> set dt equal to simulation time step
            # if not, pid takes value as real time step
            u = pid(x[0], dt=t_step)
        elif controller == "LQR":
            pass
        y = odeint(model, x, ts, args=(u,))
        x = y[-1, :]

        x_hist = np.vstack((x_hist, x))
        u_hist = np.vstack((u_hist, u))
        if t + t_step == t_end:
            break
    return x_hist, u_hist


# Initial value and simulation time setting
x0 = np.deg2rad([0, 0.1])
x_ref = np.deg2rad(1)
t_end = 100
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# PID controller setting
Kp = 15
Ki = 3
Kd = 20
pid = PID(Kp, Ki, Kd, setpoint=x_ref)

# Do simulation
x_hist, _ = sim(t_end, t_step, model_pendulum, x0, controller="PID")

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, np.rad2deg(x_hist[:, 0]), 'b-', linewidth=1)
plt.plot(tspan, np.rad2deg(x_ref) * np.ones(len(tspan)), 'k--', linewidth=1)
plt.xlim([tspan[0], tspan[-1]])
plt.ylim([-2, 2])
plt.ylabel('Theta (deg)')
plt.legend(('Simulation', 'Reference'))

plt.subplot(2, 1, 2)
plt.plot(tspan, np.rad2deg(x_hist[:, 1]), 'r-', linewidth=1, label='Simulation')
plt.plot(tspan, np.zeros(len(tspan)), 'k--', linewidth=1)
plt.xlim([tspan[0], tspan[-1]])
plt.ylim([-2, 2])
plt.ylabel('Angular Velocity (deg/s)')
plt.xlabel('Time (sec)')
plt.legend()

plt.show()
