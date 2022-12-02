import numpy as np

from model.scalar_temp import dc_motor
from model.actuator import Actuator
from sim.sim_IRL_onpolicy import Sim as Sim_on_policy_Kleinmann
from sim.sim_IRL_offpolicy import Sim as Sim_off_policy_Kleinmann
from sim.sim_IRL_temp import Sim as Sim_on_policy_IRL
from utils import *
from control import lqr

# np.random.seed(1)
# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
# x0 = np.array([[4],
#                [2]])
# x0 = np.array([[3],
#                [-3]])
x0 = np.array([[-3],
               [-5]])
x_ref = np.array([[0],
                  [0]])

model = dc_motor(x0=x0, x_ref=x_ref)
dyn = model.dynamics
actuator = Actuator()
u_constraint = np.array([[-20, 20]])
agent = "1"   # 1."on-IRL" 2."on-Kleinmann", 3."off-Kleinmann"

scaler = np.diag([1])
shift = np.array([[0]])

t_end = 5
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# Do simulation
if agent == "1":
    sim = Sim_on_policy_IRL(actuator=actuator, model=model)
    x_hist, u_hist, w_hist, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, e_shift=shift, e_scaler=scaler, clipping=u_constraint, iteration='vi', tol=1e0)
if agent == "2":
    sim = Sim_on_policy_Kleinmann(actuator=actuator, model=model)
    x_hist, u_hist, P_list, K_list, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, e_shift=shift, e_scaler=scaler, clipping=u_constraint, tol=5e-1)
elif agent == "3":
    sim = Sim_off_policy_Kleinmann(actuator=actuator, model=model)
    x_hist, u_hist, P_list, K_list, cond_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, u0_shift=shift, u0_scaler=scaler, clipping=u_constraint, tol=5e-1)

if agent == "1":
    plot(x_hist, u_hist, tspan, model.x_ref, [0], type='plot', x_shape=[2, 1], u_shape=[1, 1], x_label=['x1', 'x2'], u_label=['u1'])
    plot_w(w_hist, tspan)
    plot_cond(cond_list)
    plt.show()
elif agent == "2" or "3":
    K_lqr, P_lqr, _ = lqr(model.A, model.B, model.Q, model.R)
    print("Norm difference of P_lqr and P_Kleinmann: {}".format(np.linalg.norm(P_list[-1] - P_lqr)))
    print("Norm difference of K_lqr and K_Kleinmann: {}".format(np.linalg.norm(K_list[-1] - K_lqr)))
    # DC-Motor (LQR results)
    # K : array([[0.00772631, 0.41694259]])
    # P : array([[0.04999624, 0.00386316],
    #            [0.00386316, 0.2084713 ]])

    plot(x_hist, u_hist, tspan, model.x_ref, [0], type='plot', x_shape=[2, 1], u_shape=[1, 1], x_label=['x1', 'x2'], u_label=['u1'])
    plot_P(P_list)
    plot_K(K_list)
    plot_cond(cond_list)
    plt.show()