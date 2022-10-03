import numpy as np
from scipy.integrate import odeint


def sim(t_end, t_step, model, dyn, x0, controller, x_ref, clipping=None, u_is_scalar=False):
    """
    Model simulation
    :param u_is_scalar: if u is scalar, dimension conversion is needed for clipping u
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :param dyn: State-space form linear model
    :param x0: Initial condition of the system
    :param controller: Three kinds of controller ; "PID", "LQR", "LQI"
    :param x_ref: Reference command of the system
    :param clipping: Whether or not giving constraint of u ; dictionary input
    :return: System variable x, u (vector form)
    """
    t = 0
    x = x0
    if np.shape(x) == ():  # if x is scalar, np.shape(x) == ()
        num_x = 1
    else:
        num_x = np.shape(x)[0]  # if x is vector, takes the # of elements
    # counting u input to reshape u_hist
    num_u = np.shape(model.B)[1]
    x_hist = []
    u_hist = []
    while True:
        if "PID" in controller.keys():
            u = controller["PID"](x[0], dt=t_step)
        elif "LQR" in controller.keys():
            u = controller["LQR"].dot(np.reshape(x - x_ref, (num_x, 1))).squeeze()
        elif "LQI" in controller.keys():
            u = controller["LQI"].dot(np.reshape(x - np.block([x_ref, np.zeros(model.C.shape[0])]), (num_x, 1))).squeeze()
        # If we want to give a constraint of u
        if clipping is not None:
            if u.shape == ():   # for scalar u
                u_is_scalar = True
                u = np.reshape(u, (1,))
            for u_i, constraint, i in zip(u, clipping, range(num_u)):
                u[i] = np.clip(u_i, constraint[0], constraint[1])  # constraint of u
        if u_is_scalar:  # for scalar u
            u = np.asscalar(u)
        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u)
        y = odeint(dyn, x, [t, t + t_step], args=(u,))
        x = y[-1, :]

        if np.isclose(t, t_end):
            break
        t = t + t_step
    return x_hist, np.reshape(u_hist, (num_u, -1), order='F').squeeze()
