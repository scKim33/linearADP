import numpy as np
from scipy.integrate import odeint
from model.f18_lon import f18_lon
from model.actuator import Actuator

def PI(x, model):
    if model.name == 'f18_lon':
        pi = np.array([x[0],
                       x[0] ** 2,
                       x[1],
                       x[1] ** 2,
                       x[2],
                       x[2] ** 2,
                       x[3],
                       x[3] ** 2,
                       ])
    elif model.name == 'dc_motor':
        pi = np.array([x[0] ** 2,
                       x[0] * x[1],
                       x[1] ** 2
                       ])
    return pi

def dV(w, x, model):
    if model.name == 'f18_lon':
        dv = np.array([w[0] + 2 * x[0] * w[1],
                       w[2] + 2 * x[1] * w[3],
                       w[4] + 2 * x[2] * w[5],
                       w[6] + 2 * x[3] * w[7],
                       ])
    elif model.name == 'dc_motor':
        dv = np.array([2 * x[0] * w[0] + x[1] * w[1],
                       2 * x[1] * w[2] + x[0] * w[1],
                       ])
    return dv

def sim_IRL(t_end, t_step, model, actuator, dyn, x0, x_ref, clipping=None, u_is_scalar=False, method=None, actuator_status=False):

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
    w = np.random.randn(len(PI(x, model)))
    x_hist = []
    u_hist = []
    w_hist = []
    while True:
        u_ctrl = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x, model)   # From the result of policy/value iteration : u = -0.5@R^-1@B.T@nabla(V)
        if actuator_status:
            u_act = np.array([u_ctrl[0], 0]) # u_act, u_act_dot in systems of ODE
            u_act = odeint(actuator.dynamics, u_act, [t, t + 0.1], args=(u_ctrl[0],))
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
        u_act_ = None
        PI_ = None
        for i in range(len(PI(x, model))): # len(PI(x)) equations are needed to find least square solutions
            # finding u after n * t_step
            u_ctrl_ = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x_, model)
            if actuator_status:
                u_act_ = np.array([u_ctrl_[0], 0])
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

            x__ = None
            x__list = None
            u__list = None
            u_ctrl__ = None
            interval = 10
            for _ in range(interval):
                x__ = y_[-1, :] if x__ is not None else x_
                u_ctrl__ = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dV(w, x__,
                                                                          model) if u_ctrl__ is not None else u_ctrl_
                if actuator_status:
                    u_act__ = np.array([u_ctrl__[0], 0])
                    u_act__ = odeint(actuator.dynamics, u_act__, [t_, t_ + 0.1 * t_step_], args=(u_ctrl__[0],))
                    u_act__ = u_act__[-1, :]
                    u_ctrl__[0] = u_act__[0]
                y_ = odeint(dyn, x__, [t_, t_ + 1 / interval * t_step_], args=(u_ctrl__,))    # To find r integral value
                x__ = y_[-1, :]
                x__list = np.vstack((x__list, x__)) if x__list is not None else x__ # interval x num_x matrix finally
                u__list = np.vstack((u__list, u_ctrl__)) if u__list is not None else u_ctrl__  # interval x num_x matrix finally
            x_sum = 2 * np.sum(x__list, axis=0) - x__list[-1, :] + x_
            u_sum = np.sum(u__list, axis=0)
            r_integral = 0.25 * t_step_ / interval * (x_sum.T @ model.Q @ x_sum) + t_step_ / interval * (u_sum.T @ model.R @ u_sum) # trapezoidal rule of integration
            x_old = x_
            x_ = y_[-1, :] + 0.1 * np.random.randn(num_x) * y_[-1, :]
            # x_ = y_[-1, :]



            if method == "PI":
                X = np.vstack((X, PI(x, model) - PI(x_, model))) if X is not None else PI(x, model) - PI(x_, model) # set of basis
                try:
                    R = np.append(R, R[-1] + r_integral)  # set of integral(xQx+uRu)
                except:
                    R = np.append(R, r_integral)
            elif method == "VI":
                PI_ = np.vstack((PI_, PI(x_, model))) if PI_ is not None else PI(x_, model)
                X = np.vstack((X, PI(x_old, model))) if X is not None else PI(x_old, model)
                R = np.append(R, r_integral)
            t_ = t_ + t_step_
        if method == "PI":
            w_temp = np.linalg.inv(X.T @ X) @ X.T @ R
        elif method == "VI":
            w_temp = np.linalg.inv(X.T @ X) @ X.T @ (R + PI_ @ w)
        # if np.max(abs(w)) < 20:
        #     w = w_temp
        w = w_temp
        # breakpoint()
        print("t : {}".format(t))
        print("Phi_bar : {}".format(X))
        print("W : {}".format(w))
        print("Condition number : {}".format(np.linalg.cond(X.T@X)))
        if np.isclose(t, t_end):
            w_hist = w_hist.reshape(-1, len(w))
            x_hist = x_hist.reshape(-1, len(x))
            u_hist = u_hist.reshape(-1, len(u_ctrl))
            break
        t = t + t_step
    return x_hist, u_hist, w_hist
