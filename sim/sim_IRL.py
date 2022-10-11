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

def sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref, clipping=None, u_is_scalar=False, method=None):

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
    while True:
        u_ctrl = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x)   # From the result of policy/value iteration : u = -0.5@R^-1@B.T@nabla(V)
        u_act = u_ctrl
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

        x_hist = np.append(x_hist, x)
        u_hist = np.append(u_hist, u_ctrl)
        w_hist = np.append(w_hist, w)

        y = odeint(dyn, x, [t, t + t_step], args=(u_ctrl,))
        x = y[-1, :]    # take x at (t + t_step)

        # Policy Improvement to update w
        X = None
        R = []
        t_ = t  # define t_ used in for loop
        t_step_ = 0.02
        x_ = x  # define x_ used in for loop
        r = 0
        u_act_ = u_act
        PI_ = None
        for i in range(len(PI(x))): # len(PI(x)) equations are needed to find least square solutions
            # finding u after n * t_step
            u_ctrl_ = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x_)
            u_act_ = u_ctrl_
            u_act_ = odeint(actuator.dynamics, u_act_, [t_, t_ + t_step_], args=(u_ctrl_[0],))
            u_act_ = u_act_[-1, :]
            u_ctrl_[0] = u_act_[0]

            if clipping is not None:
                if u_ctrl_.shape == ():
                    u_is_scalar = True
                    u_ctrl_ = np.reshape(u_ctrl_, (1,))
                for u_i, constraint, j in zip(u_ctrl_, clipping, range(num_u)):
                    u_ctrl_[j] = np.clip(u_i, constraint[0], constraint[1])
            if u_is_scalar:
                u_ctrl_ = np.asscalar(u_ctrl_)
            y_ = odeint(dyn, x_, [t_, t_ + t_step_], args=(u_ctrl_,))
            x_old = x_
            x_ = y_[-1, :]

            if method == "PI":
                r = r + t_step_ * (x_.T @ model.Q @ x_ + u_ctrl_.T @ model.R @ u_ctrl_)  # integral of xQx+uRu
                X = np.vstack((X, PI(x) - PI(x_))) if X is not None else PI(x) - PI(x_) # set of basis
                R = np.append(R, r)  # set of integral(xQx+uRu)
            elif method == "VI":
                r = t_step_ * (x_.T @ model.Q @ x_ + u_ctrl_.T @ model.R @ u_ctrl_)
                PI_ = np.vstack((PI_, PI(x_))) if PI_ is not None else PI(x_)
                X = np.vstack((X, PI(x_old))) if X is not None else PI(x_old)
                R = np.append(R, r)
            t_ = t_ + t_step_
        if method == "PI":
            w = np.linalg.inv(X.T @ X) @ X.T @ R
        elif method == "VI":
            w = np.linalg.inv(X.T @ X) @ X.T @ (R + PI_ @ w)

        if np.isclose(t, t_end):
            w_hist = w_hist.reshape(-1, len(w))
            x_hist = x_hist.reshape(-1, len(x))
            u_hist = u_hist.reshape(-1, len(u_ctrl))
            break
        t = t + t_step
    return x_hist, u_hist, w_hist
