import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from control import lqr

from model.scalar import antenna, pendulum
from sim import sim

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = None
x_ref = None
u_constraint = np.array([[-20, 20]])
model = pendulum(x0=x0, x_ref=x_ref)
t_end = 50
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)
agent = "LQR" # choose a controller from ["PID", "LQR", "LQI"]

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
else:
    raise ValueError("Invalid agent name")

# Do simulation
x_hist, u_hist = sim(t_end, t_step, model, dyn, x0, controller=ctrl, x_ref=model.x_ref, clipping=u_constraint)
x_hist = x_hist.reshape(len(tspan), len(x0))

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, np.rad2deg(x_hist[:, 0]), 'k-', linewidth=1.2)
plt.plot(tspan, np.rad2deg(model.x_ref[0]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.ylim([-2 * np.rad2deg(np.abs(model.x_ref[0])), 2 * np.rad2deg(np.abs(model.x_ref[0]))]) # x_ref changes at default setting
plt.grid()
plt.ylabel(r'$\theta$ (deg)')
plt.title('State trajectory')
plt.legend(('State', 'Reference'))

plt.subplot(2, 1, 2)
plt.plot(tspan, np.rad2deg(x_hist[:, 1]), 'k-', linewidth=1.2, label='State')
plt.plot(tspan, np.rad2deg(model.x_ref[1]) * np.ones(len(tspan)), 'r--', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.ylim([-2, 2])
plt.grid()
plt.ylabel(r'$\omega$ (deg / s)')
plt.xlabel('Time (sec)')
plt.legend(('State', 'Reference'))

plt.figure()
plt.plot(tspan, u_hist, 'b-', linewidth=1.2)
plt.xlim([tspan[0], tspan[-1]])
plt.grid()
plt.ylabel(r'Torque (N$\cdot$m)')
plt.title('Control trajectory')

plt.show()