import numpy as np

from model.f18_lon import f18_lon
from model.actuator import Actuator
from sim.sim_onpolicy_Kleinmann import Sim as Sim_on_policy_Kleinmann
from sim.sim_offpolicy_Kleinmann import Sim as Sim_off_policy_Kleinmann
from sim.sim_IRL import Sim as Sim_on_policy_IRL
from utils import *
from control import lqr

# np.random.seed(0)
# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
x0 = np.array([[177.02],
               [np.deg2rad(3.431)],
               [np.deg2rad(-1.09*1e-2)],
               [np.deg2rad(5.8*1e-3)]])\
     - f18_lon().x_trim.reshape((4, 1)) # x_0 setting in progress report 1
# x0 = np.array([[187.02],
#                [np.deg2rad(0.41)],
#                [np.deg2rad(9.09*1e-3)],
#                [np.deg2rad(1.8*1e-2)]])\
#      - f18_lon().x_trim.reshape((4, 1))
# x0 = np.array([[207.02],
#                [np.deg2rad(1.31)],
#                [np.deg2rad(1.09*1e-2)],
#                [np.deg2rad(-1.8*1e-2)]])\
#      - f18_lon().x_trim.reshape((4, 1))
x_ref = None

model = f18_lon(x0=x0, x_ref=x_ref)
dyn = model.dynamics
actuator = Actuator()
u_constraint = np.array([[0, 1],
                         [np.deg2rad(-20), np.deg2rad(20)]])
agent = "1"   # 1."on-IRL" 2."on-Kleinmann", 3."off-Kleinmann"

# scaler = np.diag([0.5, 0.3])
# shift = np.array([[0.5], [0]])
scaler = np.diag([0.1, 0.01])
shift = np.array([[0], [0.00]])


t_end = 50
t_step = 0.02
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# Do simulation
if agent == "1":
    sim = Sim_on_policy_IRL(actuator=None, model=model)
    x_hist, u_hist, w_hist, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, e_shift=shift, e_scaler=scaler, clipping=None, iteration='pi', tol=1e0)
if agent == "2":
    sim = Sim_on_policy_Kleinmann(actuator=None, model=model)
    x_hist, u_hist, P_list, K_list, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, e_shift=shift, e_scaler=scaler, clipping=None, tol=1e0)
elif agent == "3":
    sim = Sim_off_policy_Kleinmann(actuator=None, model=model)
    x_hist, u_hist, P_list, K_list, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, u0_shift=shift, u0_scaler=scaler, clipping=None, tol=1e0)

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
    plot_w(w_hist, tspan)
    plot_cond(cond_list)
    plt.show()
elif agent == "2" or "3":
    K_lqr, P_lqr, _ = lqr(model.A, model.B, model.Q, model.R)
    print("Norm difference of P_lqr and P_Kleinmann: {}".format(np.linalg.norm(P_list[-1] - P_lqr)))
    print("Norm difference of K_lqr and K_Kleinmann: {}".format(np.linalg.norm(K_list[-1] - K_lqr)))
    # F-18 (LQR results)
    # K : array([[1.29569582e-02, -1.59486715e-01, 2.80573531e-03, 4.45661101e-02],
    #            [1.93568908e-04, -1.23264286e-02, -1.45409991e-04, -9.48189572e-03]])
    # P : array([[1.47251758e-01, -1.81420801e+00, 3.18544776e-02, 5.04540680e-01],
    #            [-1.81420801e+00, 5.27191154e+02, 3.99874533e+00, 5.04121432e+02],
    #            [3.18544776e-02, 3.99874533e+00, 8.96117708e-02, 4.60378587e+00],
    #            [5.04540680e-01, 5.04121432e+02, 4.60378587e+00, 5.18369240e+02]])

    plot(x_hist, u_hist, tspan, x_ref_for_plot, u_ref_for_plot, type='plot', x_shape=[2, 2], u_shape=[2, 1], x_label=['$V$ (m / s)', '$\\alpha$ (deg)', '$q$ (deg / s)', '$\gamma$ (deg)'], u_label=['$\delta_T$', '$\delta_e$ (deg)'])
    plot_P(P_list)
    plot_K(K_list)
    plot_cond(cond_list)
    plt.show()