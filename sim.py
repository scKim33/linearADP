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
    if np.shape(x) == ():  # if x is scalar (dim == 0)
        num_x = 1
    else:
        num_x = np.shape(x)[0]  # if x is vector (dim != 0)
    x_hist = []
    u_hist = []
    while True:
        if "PID" in controller.keys():
            u = controller["PID"](x[0], dt=t_step)
        elif "LQR" in controller.keys():
            u = controller["LQR"].dot(np.reshape(x - x_ref, (num_x, 1))).squeeze()
        elif "LQI" in controller.keys():
            u = controller["LQI"].dot(np.reshape(x - np.block([x_ref, np.zeros(len(x_ref))]), (num_x, 1))).squeeze() #
        u = np.clip(u, np.deg2rad(-20), np.deg2rad(20)) # constraint of u
        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u)
        y = odeint(dyn, x, [t, t + t_step], args=(u,))
        x = y[-1, :]

        if np.isclose(t, t_end):
            # counting u input to reshape u_hist
            if np.shape(u) == ():   # if u is scalar (dim == 0)
                num_u = 1
            else:
                num_u = np.shape(u)[0]  # if u is vector (dim != 0)
            break
        t = t + t_step
    return x_hist, np.reshape(u_hist, (num_u, -1), order='F').squeeze()
