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
            u = controller["LQR"].dot(x - x_ref.squeeze()).squeeze()
        elif "LQI" in controller.keys():
            u = controller["LQI"].dot(x - x_ref.squeeze()).squeeze()
        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u)
        y = odeint(dyn, x, [t, t + t_step], args=(u,))
        x = y[-1, :]

        if np.isclose(t, t_end):
            num_u = np.shape(u)[0]
            break
        t = t + t_step
    return x_hist, np.reshape(u_hist, (num_u, -1), order='F')