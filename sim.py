import numpy as np
from scipy.integrate import odeint


def sim(t_end, t_step, dyn, x0, controller, x_ref):
    """
    Model simulation
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :param dyn: State-space form linear model
    :param x0: Initial condition of the system
    :param controller: Three kinds of controller ; "PID", "LQR", "LQI"
    :param x_ref: Reference command of the system
    :return: System variable x, u (vector form)
    """
    t = 0
    x = x0
    x_hist = []
    u_hist = []
    while True:
        if "PID" in controller.keys():
            u = controller["PID"](x[0], dt=t_step)
        elif "LQR" in controller.keys():
            u = controller["LQR"].dot(x - np.array([x_ref, 0])).squeeze()
        elif "LQI" in controller.keys():
            u = controller["LQI"].dot(x - np.array([x_ref, 0, 0])).squeeze()
        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u)
        y = odeint(dyn, x, [t, t + t_step], args=(u,))
        x = y[-1, :]

        if np.isclose(t, t_end):
            break
        t = t + t_step
    return x_hist, u_hist