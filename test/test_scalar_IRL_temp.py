import numpy as np

from model.scalar_temp import dc_motor
from model.actuator import Actuator
from sim.sim_IRL_onpolicy import Sim as Sim_on_policy
from sim.sim_IRL_offpolicy import Sim as Sim_off_policy
from utils import plot, plot_K, plot_P

# Initial value and simulation time setting
# If needed, fill x0, x_ref, or other matrices
# x0 = np.array([[4],
#                [2]])
# x0 = np.array([[-3],
#                [-5]])
x0 = np.array([[3],
               [-3]])
x_ref = np.array([[0],
                  [0]])

model = dc_motor(x0=x0, x_ref=x_ref)
dyn = model.dynamics
actuator = Actuator()
u_constraint = np.array([[-20, 20]])
agent = "on"   # 1."on", 2."off"

t_end = 10
t_step = 0.1
tspan = np.linspace(0, t_end, int(t_end / t_step) + 1)

# Do simulation
if agent == "on":
    sim = Sim_on_policy(actuator=actuator, model=model)
    x_hist, u_hist, P_list, K_list = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint, constraint_P=10, constraint_K=10, tol=1e-3)
elif agent == "off":
    sim = Sim_off_policy(actuator=actuator, model=model)
    x_hist, u_hist = sim.sim(t_end, t_step, dyn, x0, x_ref=model.x_ref, clipping=u_constraint)

plot(x_hist, u_hist, tspan, model.x_ref, [0], type='plot', x_shape=[2,1], u_shape=[1,1], x_label=['x1', 'x2'], u_label=['u1'])
plot_P(P_list)
plot_K(K_list)