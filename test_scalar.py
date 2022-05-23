import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from control import lqr

from scalar import model_pendulum, model_antenna, model_f18_lat
from sim import sim
from scipy.io import loadmat

# Initial value and simulation time setting
# x0 = np.deg2rad([0, 0.1])
# x_ref = np.deg2rad([1, 0])
# model = model_pendulum(x_ref)
x0 = np.array([190, 0.005, 0.001, 0.001])
x_ref = loadmat('dat/f18_lin_data.mat')['x_trim_lon'].squeeze()
model = model_f18_lat(x_ref)
t_end = 50
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)
agent = "LQI" # choose a controller from ["PID", "LQR", "LQI"]

if agent == "PID":
    # PID controller setting
    Kp = 1
    Ki = 0.1
    Kd = 1
    ctrl = {"PID": PID(Kp, Ki, Kd, setpoint=x_ref)}
    # simulation condition -> set dt equal to simulation time step
    # if not, pid takes value as real time step
    dyn = model.dynamics
elif agent == "LQR":
    # LQR controller setting
    # Have to adjust the size of Q with same size of dynamic matrix A
    # Have to adjust the size of R with same size of u
    # Q = np.diag([1e5, 1])
    # R = np.diag([1])
    Q = np.diag([1, 1, 1, 1])
    R = np.diag([1e6, 1e6])
    K, _, _ = lqr(model.A, model.B, Q, R)
    ctrl = {"LQR": -K}
    dyn = model.dynamics
elif agent == "LQI":
    # LQI controller setting
    # Have to adjust the size of Qa with same size of dynamic matrix Aa
    # Have to adjust the size of Ra with same size of u
    # Qa = np.diag([100, 10, 100, 100])
    # Ra = np.diag([1])
    Qa = np.diag([1, 10, 10, 10, 10, 10, 10, 10])
    Ra = np.diag([1e6, 1e6])
    Ka, _, _ = lqr(model.Aa, model.Ba, Qa, Ra)
    ctrl = {"LQI": -Ka}
    x0 = np.append(x0, -x_ref)
    dyn = model.aug_dynamics
else:
    raise ValueError("Invalid agent name")

# Do simulation
x_hist, u_hist = sim(t_end, t_step, dyn, x0, controller=ctrl, x_ref=x_ref)
x_hist = x_hist.reshape(len(tspan), len(x0))

# # Plot the results
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(tspan, np.rad2deg(x_hist[:, 0]), 'k-', linewidth=1.2)
# plt.plot(tspan, np.rad2deg(x_ref[0]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
# plt.grid()
# plt.ylabel('Theta (deg)')
# plt.title('State trajectory')
# plt.legend(('State', 'Reference'))
#
# plt.subplot(2, 1, 2)
# plt.plot(tspan, np.rad2deg(x_hist[:, 1]), 'k-', linewidth=1.2, label='State')
# plt.plot(tspan, np.rad2deg(x_ref[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
# plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
# plt.xlabel('Time (sec)')
# plt.legend()
#
# plt.figure()
# plt.plot(tspan, np.rad2deg(u_hist), 'b-', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.grid()
# plt.ylabel('Torque')
# plt.title('Control trajectory')
#
# plt.show()

# Plot the results
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(tspan, x_hist[:, 0], 'k-', linewidth=1.2)
plt.plot(tspan, model.x_trim[0] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Theta (deg)')
plt.ylabel('u (m/s)')
plt.xlabel('Time (sec)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 2)
plt.plot(tspan, np.rad2deg(x_hist[:, 1]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('alpha (deg)')
plt.xlabel('Time (sec)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 3)
plt.plot(tspan, np.rad2deg(x_hist[:, 2]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[2]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('q (deg/s)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 4)
plt.plot(tspan, np.rad2deg(x_hist[:, 3]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[3]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('theta (deg)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, np.rad2deg(u_hist[0]), 'b-', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel('Control Input 1 (deg)')
plt.title('Control trajectory')

plt.subplot(2, 1, 2)
plt.plot(tspan, np.rad2deg(u_hist[1]), 'b-', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel('Control Input 2 (deg)')

plt.show()