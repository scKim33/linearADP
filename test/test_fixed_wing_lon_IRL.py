import numpy as np

from model.fixed_wing_lon import fixed_wing_lon
from model.actuator import Actuator
from sim.sim_IRL_onpolicy import Sim as Sim_on_policy_Kleinmann
from sim.sim_IRL_offpolicy import Sim as Sim_off_policy_Kleinmann
from sim.sim_IRL_temp import Sim as Sim_on_policy_IRL
from utils import plot, plot_K, plot_P, plot_w
from control import lqr

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = np.array([[0.02],
               [np.deg2rad(1.431)],
               [np.deg2rad(-1.09*1e-2)],
               [np.deg2rad(5.8*1e-3)]])\
     - fixed_wing_lon().x_trim.reshape((4, 1))
x_ref = None

model = fixed_wing_lon(x0=x0, x_ref=x_ref)
dyn = model.dynamics
actuator = Actuator()
u_constraint = np.array([[0 - model.u_trim[0], 1 - model.u_trim[0]],
                         [np.deg2rad(-20), np.deg2rad(20)]])
agent = "3"   # 1."on-IRL" 2."on-Kleinmann", 3."off-Kleinmann"

t_end = 50
t_step = 0.02
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# Do simulation
if agent == "1":
    sim = Sim_on_policy_IRL(actuator=actuator, model=model)
    x_hist, u_hist, w_hist = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, iteration='pi', tol=1e3)
if agent == "2":
    sim = Sim_on_policy_Kleinmann(actuator=actuator, model=model)
    x_hist, u_hist, P_list, K_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, constraint_P=1e5, constraint_K=1e3, tol=5e2)
elif agent == "3":
    sim = Sim_off_policy_Kleinmann(actuator=actuator, model=model)
    x_hist, u_hist, P_list, K_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, constraint_P=1e6, constraint_K=1e3, tol=1e4)

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
u_hist[1, :] = np.rad2deg(u_hist[1, :])

if agent == "1":
    plot(x_hist, u_hist, tspan, x_ref_for_plot, u_ref_for_plot, type='plot', x_shape=[2, 2], u_shape=[2, 1], x_label=['$V$ (m / s)', '$\\alpha$ (deg)', '$q$ (deg / s)', '$\gamma$ (deg)'], u_label=['$\delta_T$', '$\delta_e$ (deg)'])
    plot_w(w_hist)
elif agent == "2" or "3":
    K_lqr, P_lqr, _ = lqr(model.A, model.B, model.Q, model.R)
    print("Norm difference of P_lqr and P_Kleinmann: {}".format(np.linalg.norm(P_list[-1] - P_lqr)))
    print("Norm difference of K_lqr and K_Kleinmann: {}".format(np.linalg.norm(K_list[-1] - K_lqr)))
    # Fixed-wing (LQR results)
    # K : array([[ 1.14267602e-01, -2.25964079e-01,  1.48497910e-01,  2.33960105e-01],
    #            [-1.13995087e-04,  8.92630120e-05, -1.58315160e-03, -5.75454352e-03]])
    # P : array([[  2.26683777,  -4.4918347 ,   2.57553486,   3.13112153],
    #            [ -4.4918347 ,  51.62423059,  -2.17686125,  39.66228074],
    #            [  2.57553486,  -2.17686125,  36.48145093, 133.50374551],
    #            [  3.13112153,  39.66228074, 133.50374551, 568.12022155]])

    plot(x_hist, u_hist, tspan, x_ref_for_plot, u_ref_for_plot, type='plot', x_shape=[2, 2], u_shape=[2, 1], x_label=['$V$ (m / s)', '$\\alpha$ (deg)', '$q$ (deg / s)', '$\gamma$ (deg)'], u_label=['$\delta_T$', '$\delta_e$ (deg)'])
    plot_P(P_list)
    plot_K(K_list)