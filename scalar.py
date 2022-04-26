import time
import matplotlib.pyplot as plt
from simple_pid import PID
import numpy as np
from scipy.integrate import odeint

def model(x, t, u, J, B):
    """
    Satellite tracking antenna
    theta: antenna angle from the origin
    :param x: State-space system parameter [theta, theta_dot]
    :param u: Control torque from the drive motor, given by PID controller
    :param J: a moment of inertia of antenna and driving parts
    :param B: a damping of the system
    :return: x_dot
    """

    # state - space specification
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = -B / J * x[1] + u / J

    return x_dot

# J = 600,000 (kg * m^2)
J = 600000
# B = 20,000 (N * m * sec)
B = 20000

# PID controller setting
theta_ref = 10
Kp = 2000
Ki = 0
Kd = 200
pid = PID(Kp, Ki, Kd, setpoint=theta_ref)

# Time step setting
t = np.linspace(0, 1000, 1001)

# Data assigning for plotting
theta = np.zeros(len(t))
angular_vel = np.zeros(len(t))
u = np.zeros(len(t))

# Initial state setting
x0 = np.zeros(2)

# loop through time steps
for i in range(len(t) - 1):
    dt = t[i+1] - t[i]
    u[i + 1] = pid(theta[i], dt=1)  # simulation condition -> set dt equal to simulation time step
                                    # if not, pid takes value as real time step
    ts = [t[i], t[i + 1]]
    y = odeint(model, x0, ts, args=(u[i + 1], J, B))
    theta[i + 1] = y[-1][0]
    angular_vel[i + 1] = y[-1][1]
    x0[0] = theta[i + 1]
    x0[1] = angular_vel[i + 1]

# Plot the results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, theta, 'b-', linewidth=1)
plt.plot(t, theta_ref * np.ones(len(t)), 'k--', linewidth=1)
plt.xlim([t[0], t[-1]])
plt.ylabel('Theta (rad)')
plt.legend(('sim', 'ref'), loc='best')

plt.subplot(3, 1, 2)
plt.plot(t, u / 1000, 'r-', linewidth=1)
plt.plot(t, np.zeros(len(t)), 'k-', linewidth=1)
plt.xlim([t[0], t[-1]])
plt.ylabel('Control Torque (kN·m)')
plt.legend(['sim'], loc='best')

plt.subplot(3, 1, 3)
plt.plot(t, angular_vel, 'k-', linewidth=1, label='sim')
plt.plot(t, np.zeros(len(t)), 'k-', linewidth=1)
plt.xlim([t[0], t[-1]])
plt.ylabel('Angular Velocity (rad/s)')
plt.xlabel('Time (sec)')
plt.legend(loc='best')

plt.show()