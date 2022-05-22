import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from control import lqr

from scalar import model_pendulum, model_antenna, model_f18_lat
from sim import sim

# Initial value and simulation time setting
x0 = np.array([160, 0.01, 0.01, 0.01])
# model = model_pendulum(x_ref)
# x_ref = np.deg2rad(1)
model = model_f18_lat()
x_ref = model.x_trim
t_end = 100
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)
agent = "LQR" # choose a controller from ["PID", "LQR", "LQI"]

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
    Q = np.diag([0.01, 0.01, 0.1, 0.1])
    R = np.diag([1e10, 1e10])
    K, _, _ = lqr(model.A, model.B, Q, R)
    ctrl = {"LQR": -K}
    dyn = model.dynamics
elif agent == "LQI":
    # LQI controller setting
    Qa = np.diag([100, 10, 100])
    Ra = np.diag([1])
    Ka, _, _ = lqr(model.Aa, model.Ba, Qa, Ra)
    ctrl = {"LQI": -Ka}
    x0 = np.append(x0, -x_ref)
    dyn = model.aug_dynamics
else:
    raise ValueError("Invalid agent name")

# Do simulation
x_hist, u_hist = sim(t_end, t_step, dyn, x0, controller=ctrl, x_ref=x_ref)
x_hist = x_hist.reshape(len(tspan), len(x0))

# Plot the results
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(tspan, x_hist[:, 0], 'k-', linewidth=1.2)
plt.plot(tspan, x_ref[0] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Theta (deg)')
plt.ylabel('u (m/s)')
plt.xlabel('Time (sec)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 2)
plt.plot(tspan, x_hist[:, 1], 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(x_ref[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('alpha (rad/s)')
plt.xlabel('Time (sec)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 3)
plt.plot(tspan, x_hist[:, 2], 'k-', linewidth=1.2)
plt.plot(tspan, x_ref[2] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('q (rad/s)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 4)
plt.plot(tspan, x_hist[:, 3], 'k-', linewidth=1.2)
plt.plot(tspan, x_ref[3] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
plt.grid()
# plt.ylabel('Angular Velocity (deg/s)')
plt.ylabel('theta (rad)')
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