import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from control import lqr

from model.f18_lon import f18_lon
from model.actuator import Actuator
from sim.sim import sim
from sim.sim_IRL import sim_IRL

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = None
x_ref = None
model = f18_lon(x0=x0, x_ref=x_ref)
u_constraint = np.array([[0 - model.u_trim[0], 1 - model.u_trim[0]],
                         [np.deg2rad(-20), np.deg2rad(20)]])
actuator = Actuator()
t_end = 50
t_step = 0.02
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)
agent = "IRL" # choose a controller from ["PID", "LQR", "LQI"]

if agent == "PID":
    # PID controller setting
    Kp = 1
    Ki = 0.1
    Kd = 1
    ctrl = {"PID": PID(Kp, Ki, Kd, setpoint=model.x_ref)}
    # simulation condition -> set dt equal to simulation time step
    # if not, pid takes value as real time step
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
else:
    raise ValueError("Invalid agent name")

# Do simulation
if agent == "IRL":
    x_hist, u_hist, w_hist = sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref=model.x_ref, clipping=u_constraint)
else:
    x_hist, u_hist = sim(t_end, t_step, model, actuator, dyn, x0, controller=ctrl, x_ref=model.x_ref, clipping=u_constraint)

# Plot the results
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(tspan, x_hist[:, 0] + model.x_trim[0], 'k-', linewidth=1.2)
plt.plot(tspan, model.x_trim[0] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$V$ (m / s)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 2)
plt.plot(tspan, np.rad2deg(x_hist[:, 1] + model.x_trim[1]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$\alpha$ (deg)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 3)
plt.plot(tspan, np.rad2deg(x_hist[:, 2] + model.x_trim[2]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[2]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$q$ (deg / s)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.subplot(2, 2, 4)
plt.plot(tspan, np.rad2deg(x_hist[:, 3] + model.x_trim[3]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_trim[3]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$\gamma$ (deg)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, u_hist[:, 0] + model.u_trim[0], 'b-', linewidth=1.2)
plt.plot(tspan, model.u_trim[0] * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$\delta_T$')
plt.title('Control trajectory')

plt.subplot(2, 1, 2)
plt.plot(tspan, np.rad2deg(u_hist[:, 1] + model.u_trim[1]), 'b-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.u_trim[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'$\delta_e$ (deg)')

if agent == "IRL":
    plt.figure()
    plt.plot(tspan, w_hist[:, 0], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 1], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 2], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 3], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 4], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 5], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 6], '--', linewidth=1.2)
    plt.plot(tspan, w_hist[:, 7], '--', linewidth=1.2)
    plt.legend(('w(0)', 'w(1)', 'w(2)', 'w(3)', 'w(4)', 'w(5)', 'w(6)', 'w(7)'))
    plt.xlim([tspan[0], tspan[-1]])
    plt.grid()
    plt.title(r'Weight of Value Function')
plt.show()
