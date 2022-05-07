import time
import matplotlib.pyplot as plt
from simple_pid import PID
import numpy as np
from scipy.integrate import odeint
from control import lqr

class model:
    def __init__(self, A, B, args):
        self.A = np.array(A)
        self.B = np.array(B)
        self.size_of_x = self.A.shape[1]
        self.size_of_u = self.B.shape[1]
        self.Q = np.eye(self.size_of_x)
        self.R = np.eye(self.size_of_u)
        self.args = tuple(args)

    def x_dot(self, x, t, u, *args):
        # reshape is needed to match dimension A @ x and B @ u
        return np.dot(self.A, x) + np.reshape(np.dot(self.B, u), -1)

## model antenna

# Satellite tracking antenna
# theta: antenna angle from the origin
# :param x: State-space system parameter [theta, theta_dot]
# :param u: Control torque from the drive motor, given by PID controller
# :param J: a moment of inertia of antenna and driving parts
# :param B: a damping of the system
# :return: x_dot

# J = 600,000 (kg * m^2)
J = 600000
# B = 20,000 (N * m * sec)
B = 20000
# System matrices
A = np.array([[0, 1],
              [0, -B / J]])
B_ = np.array([[0],
               [1 / J]])
antenna = model(A, B_, args=(J, B))
antenna.Q = np.diag([10, 1])
antenna.R = np.eye(1)

## model pendulum

# Pendulum system with torque
# theta: Angle of the pendulum from the gravity direction
# :param x: State-space system parameter [theta, theta_dot]
# :param u: Control torque about the pivot point, given by PID controller
# :param g: Acceleration of gravity
# :param l: Pendulum length
# :return: x_dot

# g = 9.81 (m / s^2)
g = 9.81
# l = (m)
l = 20
# m = (kg)
m = 0.1

# System matrices
A = np.array([[0, 1],
              [-g / l, 0]])
B = np.array([[0],
              [1 / (m * np.power(l, 2))]])
pendulum = model(A, B, args=(g, l, m))
pendulum.Q = np.diag([10, 1])
pendulum.R = np.eye(1)

def Sim(t_end, t_step, model, x0, controller):
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

    t = np.linspace(0, t_end, int(t_end / t_step) + 1)
    x = np.zeros((len(x0), len(t)))
    u = np.zeros(len(t))
    if controller == "PID":
        for i in range(len(t) - 1):
            u[i + 1] = pid(x[0, i], dt=t_step)
            # simulation condition -> set dt equal to simulation time step
            # if not, pid takes value as real time step
            ts = [t[i], t[i + 1]]
            y = odeint(model.x_dot, x0, ts, args=((u[i + 1],) + model.args))
            x[:, i + 1] = y[-1, :]
            x0 = x[:, i + 1]
    elif controller == "LQR":
        K, _, _ = lqr(model.A, model.B, model.Q, model.R)
        print(K)
        for i in range(len(t) - 1):
            u[i + 1] = -K @ (x[:, i] - np.reshape(x_ref, -1))
            ts = [t[i], t[i + 1]]
            y = odeint(model.x_dot, x0, ts, args=((u[i + 1],) + model.args))
            x[:, i + 1] = y[-1, :]
            x0 = x[:, i + 1]
    return x


# Initial value and simulation time setting
x0 = np.array([0, 0.01])
t_end = 100
t_step = 0.1
t = np.linspace(0, t_end, int(t_end / t_step) + 1)

# PID controller setting
x_ref = np.array([[1 * np.pi / 180],
                  [0]]) # Rad to Degree
Kp = 15
Ki = 7
Kd = 10
pid = PID(Kp, Ki, Kd, setpoint=x_ref[0])

# Do simulation
# x = Sim(t_end, t_step, pendulum, x0, controller="PID")
x = Sim(t_end, t_step, pendulum, x0, controller="LQR")

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x[0], 'b-', linewidth=1)
plt.plot(t, x_ref[0] * np.ones(len(t)), 'k--', linewidth=1)
plt.xlim([t[0], t[-1]])
plt.ylim([-0.02, 0.04])
plt.ylabel('Theta (rad)')
plt.legend(('Simulation', 'Reference'))

plt.subplot(2, 1, 2)
plt.plot(t, x[1], 'r-', linewidth=1, label='Simulation')
plt.plot(t, x_ref[1] * np.ones(len(t)), 'k--', linewidth=1)
plt.xlim([t[0], t[-1]])
plt.ylim([-0.03, 0.03])
plt.ylabel('Angular Velocity (rad/s)')
plt.xlabel('Time (sec)')
plt.legend()

plt.show()
