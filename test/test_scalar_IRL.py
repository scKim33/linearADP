import numpy as np
import matplotlib.pyplot as plt

from model.scalar import pendulum, dc_motor
from model.actuator import Actuator
from sim.sim_IRL import sim_IRL
from sim.sim_IRL_onpolicy import sim_IRL_onpolicy

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = np.array([4, 2])
# x0 = None
x_ref = None
u_constraint = np.array([[-20, 20]])
model = dc_motor(x0=x0, x_ref=x_ref)

actuator = Actuator()
t_end = 3
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# IRL controller setting
Q = model.Q
R = model.R
x0 = model.x0
dyn = model.dynamics
method = "PI"

# Do simulation
x_hist, u_hist, w_hist = sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, method=method, actuator_status=False)

# Plot the results
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(tspan, np.rad2deg(x_hist[:, 0]), 'k-', linewidth=1.2)
# plt.plot(tspan, np.rad2deg(model.x_ref[0]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2 * np.rad2deg(np.abs(model.x_ref[0])), 2 * np.rad2deg(np.abs(model.x_ref[0]))]) # x_ref changes at default setting
# plt.grid()
# plt.ylabel(r'$\theta$ (deg)')
# plt.title('State trajectory')
# plt.legend(('State', 'Reference'))
#
# plt.subplot(2, 1, 2)
# plt.plot(tspan, np.rad2deg(x_hist[:, 1]), 'k-', linewidth=1.2, label='State')
# plt.plot(tspan, np.rad2deg(model.x_ref[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 2])
# plt.grid()
# plt.ylabel(r'$\omega$ (deg / s)')
# plt.xlabel('Time (sec)')
# plt.legend(('State', 'Reference'))
#
# plt.figure()
# plt.plot(tspan, u_hist, 'b-', linewidth=1.2)
# plt.xlim([tspan[0], tspan[-1]])
# plt.grid()
# plt.ylabel(r'Torque (N$\cdot$m)')
# plt.title('Control trajectory')
#
# plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(tspan, x_hist[:, 0], s=15, c='k', marker='x', linewidth=1.2)
plt.plot(tspan, model.x_ref[0] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 6])
plt.grid()
plt.ylabel(r'x1')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 1, 2)
plt.scatter(tspan, x_hist[:, 1], s=15, c='k', marker='x', linewidth=1.2)
plt.plot(tspan, model.x_ref[1] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-2, 6])
plt.grid()
plt.ylabel(r'x2')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.figure()
plt.scatter(tspan, u_hist, s=15, c='b', marker='x', linewidth=1.2)
plt.plot(tspan, np.zeros(len(tspan)), 'r--', linewidth=1.2, label='reference')
plt.legend()
plt.xlim([tspan[0], tspan[-1]])
# plt.ylim([-1, 0.2])
plt.grid()
plt.ylabel(r'u')
plt.title('Control trajectory')

plt.figure()
for i in range(len(w_hist[0, :])):
    plt.plot(tspan, w_hist[:, i], 'x', linewidth=1.2, label='w[{}]'.format(i))
plt.xlim([tspan[0], tspan[-1]])
plt.legend()
plt.grid()
plt.title(r'Weight of Value Function')
plt.show()