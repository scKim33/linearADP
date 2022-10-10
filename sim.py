import numpy as np
from scipy.integrate import odeint
from model.actuator import Actuator


def sim(t_end, t_step, model, actuator, dyn, x0, controller, x_ref, clipping=None, u_is_scalar=False):

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
    u_act = None
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
            u_ctrl = controller["PID"](x[0], dt=t_step)
        elif "LQR" in controller.keys():
            u_ctrl = controller["LQR"].dot(np.reshape(x - x_ref, (num_x, 1))).squeeze()
        elif "LQI" in controller.keys():
            u_ctrl = controller["LQI"].dot(np.reshape(x - np.block([x_ref, np.zeros(model.C.shape[0])]), (num_x, 1))).squeeze()
        if u_act is None:
            u_act = np.array([u_ctrl[0], 0])   # set u_actuator initial condition at first time step
        u_act = odeint(actuator.dynamics, u_act, [t, t + t_step], args=(u_ctrl[0],))
        u_act = u_act[-1, :]    # take u_act at (t + t_step)
        u_ctrl[0] = u_act[0]    # only considering throttle actuator effect

        # If we want to give a constraint of u
        if clipping is not None:
            if u_ctrl.shape == ():   # for scalar u
                u_is_scalar = True
                u_ctrl = np.reshape(u_ctrl, (1,))
            for u_i, constraint, i in zip(u_ctrl, clipping, range(num_u)):
                u_ctrl[i] = np.clip(u_i, constraint[0], constraint[1])  # constraint of u
        if u_is_scalar:  # for scalar u
            u_ctrl = np.asscalar(u_ctrl)

        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u_ctrl)
        y = odeint(dyn, x, [t, t + t_step], args=(u_ctrl,))
        x = y[-1, :]    # take x at (t + t_step)

        if np.isclose(t, t_end):
            break
        t = t + t_step
    return x_hist, np.reshape(u_hist, (num_u, -1), order='F').squeeze()
