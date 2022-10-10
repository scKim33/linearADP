import numpy as np
from scipy.integrate import odeint
from model.f18_lon import f18_lon
from model.actuator import Actuator

def PI(x):
    pi = np.array([x[0],
                   x[0] ** 2,
                   x[1],
                   x[1] ** 2,
                   x[2],
                   x[2] ** 2,
                   x[3],
                   x[3] ** 2,
                   ])
    return pi

def dV(w, x):
    dv = np.array([w[0] + 2 * x[0] * w[1],
                   w[2] + 2 * x[1] * w[3],
                   w[4] + 2 * x[2] * w[5],
                   w[6] + 2 * x[3] * w[7],
                   ])
    return dv

def sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref, clipping=None, u_is_scalar=False):

    """
    Model simulation
    :param u_is_scalar: if u is scalar, dimension conversion is needed for clipping u
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :param dyn: State-space form linear model
    :param x0: Initial condition of the system
    :param x_ref: Reference command of the system
    :param clipping: Whether or not giving constraint of u ; dictionary input
    :return: System variable x, u, parameter of value function w (vector form)
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
    w = np.random.randn(8)
    x_hist = []
    u_hist = []
    w_hist = []
    count = 0
    X = None
    R = []
    while True:
        # breakpoint()
        u_ctrl = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x)   # From the result of policy/value iteration : u = -0.5@R^-1@B.T@nabla(V)

        if u_act is None:
            u_act = np.array([u_ctrl[0], 0])   # set u_actuator initial condition at first time step
        u_act = odeint(actuator.dynamics, u_act, [t, t + t_step], args=(u_ctrl[0],))
        u_act = u_act[-1, :]    # take u_act at (t + t_step)

        u_ctrl[0] = u_act[0]  # only considering throttle actuator effect

        # If we want to give a constraint of u
        if clipping is not None:
            if u_ctrl.shape == ():   # for scalar u
                u_is_scalar = True
                u_ctrl = np.reshape(u_ctrl, (1,))
            for u_i, constraint, i in zip(u_ctrl, clipping, range(num_u)):
                u_ctrl[i] = np.clip(u_i, constraint[0], constraint[1])  # constraint of u
        if u_is_scalar:  # for scalar u
            u_ctrl = np.asscalar(u_ctrl)

        r = t_step * (x.T @ model.Q @ x + u_ctrl.T @ model.R @ u_ctrl)  # integral of xQx+uRu

        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u_ctrl)
        w_hist = np.append(w_hist, w)

        y = odeint(dyn, x, [t, t + t_step], args=(u_ctrl,))
        x = y[-1, :]    # take x at (t + t_step)

        count += 1
        R = np.append(R, r) # set of integral(xQx+uRu)
        X = np.vstack((X, PI(x))) if X is not None else PI(x) # set of basis
        if count >= len(PI(x)):
            if True:    #np.linalg.cond(X.T@X) < 1e3: # weight update after some x updates
                w = np.linalg.inv(X.T @ X) @ X.T @ (R + X @ w)
                count = 0
                X = None
                R = []

        if np.isclose(t, t_end):
            w_hist = w_hist.reshape(-1, len(w))
            x_hist = x_hist.reshape(-1, len(x))
            u_hist = u_hist.reshape(-1, len(u_ctrl))
            break
        t = t + t_step
    return x_hist, u_hist, w_hist
