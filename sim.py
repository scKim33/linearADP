import numpy as np
from scipy.integrate import odeint


def sim(t_end, t_step, model, x0, controller):
    """
    Linear model simulation
    :param model: State-space form linear model
    :param controller: Two kinds of controller ; "PID" or "LQR"
    :param x0: Initial condition of the system
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :return: System variable x (vector form)
    """
    t = 0
    x = x0
    x_hist = []
    u_hist = []
    while True:
        if "PID" in controller.keys():
            u = controller["PID"](x[0], dt=t_step)
        elif "LQR" in controller.keys():
            pass
        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u)
        y = odeint(model, x, [t, t + t_step], args=(u,))
        x = y[-1, :]

        if np.isclose(t, t_end):
            break
        t = t + t_step
    return x_hist, u_hist
