import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from control import lqr

from model.scalar import pendulum, dc_motor
from model.actuator import Actuator
from sim.sim import sim
from sim.sim_IRL import sim_IRL

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
agent = "IRL" # choose a controller from ["PID", "LQR", "LQI"]

if agent == "PID":
    # PID controller setting
    Kp = 10
    Ki = 1
    Kd = 10
    ctrl = {"PID": PID(Kp, Ki, Kd, setpoint=model.x_ref)}
    # simulation condition -> set dt equal to simulation time step
    # if not, pid takes value as real time step
    x0 = model.x0
    dyn = model.dynamics
elif agent == "LQR":
    # LQR controller setting
    Q = model.Q
    R = model.R
    K, _, _ = lqr(model.A, model.B, Q, R)
    ctrl = {"LQR": -K}
    x0 = model.x0
    dyn = model.dynamics
elif agent == "LQI":
    # LQI controller setting
    Qa = model.Qa
    Ra = model.Ra
    Ka, _, _ = lqr(model.Aa, model.Ba, Qa, Ra)
    ctrl = {"LQI": -Ka}
    x0 = np.append(model.x0, - model.C @ model.x_ref) # x0 dim : (n,), x_ref dim : (# of C matrix row,) -> x0 dim : (n + C mat row,)
    dyn = model.aug_dynamics
elif agent == "IRL":
    # IRL controller setting
    Q = model.Q
    R = model.R
    x0 = model.x0
    dyn = model.dynamics
    method = "PI"
else:
    raise ValueError("Invalid agent name")

# Do simulation
if agent == "IRL":
    x_hist, u_hist, w_hist = sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, method=method, actuator_status=False)
else:
    x_hist, u_hist = sim(t_end, t_step, model, actuator, dyn, x0, controller=ctrl, x_ref=model.x_ref, clipping=u_constraint, actuator_status=False)

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

if agent == "IRL":
    plt.figure()
    for i in range(len(w_hist[0, :])):
        plt.plot(tspan, w_hist[:, i], 'x', linewidth=1.2, label='w[{}]'.format(i))
    plt.xlim([tspan[0], tspan[-1]])
    plt.legend()
    plt.grid()
    plt.title(r'Weight of Value Function')
plt.show()