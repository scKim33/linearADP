import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID

from scalar import model_pendulum, model_antenna
from sim import sim


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
pid = {"PID": PID(Kp, Ki, Kd, setpoint=x_ref)}
# simulation condition -> set dt equal to simulation time step
# if not, pid takes value as real time step
# lqr = {"LQR": }

# Do simulation
x_hist, _ = sim(t_end, t_step, model_pendulum, x0, controller=pid)
x_hist = x_hist.reshape(len(tspan), len(x0))

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
