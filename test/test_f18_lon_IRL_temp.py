import numpy as np

from model.f18_lon import f18_lon
from model.actuator import Actuator
from sim.sim_IRL_onpolicy import Sim
from utils import plot

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = np.array([[177.02],
               [np.deg2rad(3.431)],
               [np.deg2rad(-1.09*1e-2)],
               [np.deg2rad(5.8*1e-3)]])\
     - f18_lon().x_trim.reshape((4, 1)) # x_0 setting in progress report 1
x_ref = None

model = f18_lon(x0=x0, x_ref=x_ref)
dyn = model.dynamics
actuator = Actuator()
u_constraint = np.array([[0 - model.u_trim[0], 1 - model.u_trim[0]],
                         [np.deg2rad(-20), np.deg2rad(20)]])

t_end = 50
t_step = 0.02
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# Do simulation
sim = Sim(actuator=actuator, model=model)
x_hist, u_hist = sim.sim_IRL_on_policy(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint)

# Plot the results
x_ref_for_plot = [model.x_trim[0],
                  np.rad2deg(model.x_trim[1]),
                  np.rad2deg(model.x_trim[2]),
                  np.rad2deg(model.x_trim[3])]
u_ref_for_plot = [model.u_trim[0],
                  np.rad2deg(model.u_trim[1])]
x_hist = x_hist + model.x_trim.reshape((4, 1))
x_hist[1:, :] = np.rad2deg(x_hist[1:, :])
u_hist = u_hist + model.u_trim.reshape((2, 1))
print(u_hist[0, :])
u_hist[1, :] = np.rad2deg(u_hist[1, :])
plot(x_hist, u_hist, tspan, x_ref_for_plot, u_ref_for_plot, type='plot', x_shape=[2, 2], u_shape=[2, 1], x_label=['$V$ (m / s)', '$\\alpha$ (deg)', '$q$ (deg / s)', '$\gamma$ (deg)'], u_label=['$\delta_T$', '$\delta_e$ (deg)'])