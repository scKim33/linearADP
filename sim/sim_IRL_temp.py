import numpy as np
from scipy.integrate import odeint
from model.f18_lon import f18_lon
from model.actuator import Actuator

class Sim:
    def __init__(self,
                 actuator=None,
                 model=None):
        self.actuator = actuator
        self.actuator_enable = True
        if self.actuator is None:
            self.actuator_enable = False
        self.model = model
        self.m = np.shape(self.model.A)[1]
        self.n = np.shape(self.model.B)[1]

        try:
            if self.model == None:
                raise
        except:
            print('Assign the model.')

    def actuator_u(self, u_ctrl, enabling_index=0, t_step=0.1):
        dyn = self.actuator.dynamics
        if self.actuator_enable:
            u_act = np.array([u_ctrl[enabling_index], 0])  # u_act, u_act_dot in systems of ODE
            u_act = odeint(dyn, u_act, [0, t_step],
                           args=(u_ctrl[enabling_index],))  # only considering enabling_index element
            u_act = u_act[-1, :]  # take u_act at t_step
            u_ctrl[0] = u_act[0]
        return u_ctrl  # same size as input u_ctrl

    def clipping_u(self, u, clipping):
        if clipping is not None:
            n = self.n
            u = u.reshape((n, 1))  # Reshaping u of dim=2
            for u_i, constraint, i in zip(u, clipping, range(n)):
                u[i, 0] = np.clip(u_i, constraint[0], constraint[1])  # constraint of u
        return u  # (n, 1)

    def pi(self, x):
        # x is given as (m, 1)
        x = x.squeeze()
        m = self.m
        num_w = int(m + m * (m + 1) / 2)
        pi = np.zeros((num_w, 1))
        for i in range(m):
            pi[i, 0] = x[i]
        el = m
        for j in range(m):
            for k in range(j, m):
                pi[el, 0] = x[j] * x[k]
                el += 1
        return pi

    def dpi_dx(self, x):
        # x is given as (m, 1)
        x = x.squeeze()
        m = self.m
        dpi_dx = np.eye(m)
        for i in range(m):
            submat = np.zeros((m - i, m))
            submat[:, i:] += x[i] * np.eye(m - i)
            for j in range(i, m):
                submat[j - i, i] += x[j]
            dpi_dx = np.vstack((dpi_dx, submat))
        return dpi_dx

    def r(self, x_list, u_list, dx):
        m = self.m
        n = self.n

        r_list = []
        for i in range(x_list.shape[1]):
            x = x_list[:, -i - 1].reshape((m, 1))
            u = u_list[:, -i - 1].reshape((n, 1))
            r_list.append(x.T @ self.model.Q @ x + u.T @ self.model.R @ u)
        r_list = np.array(r_list).squeeze()
        r = np.trapz(r_list, dx=dx)
        return r.reshape((1, 1))

    def policy_iteration(self, dyn, x, w, clipping, tol):
        m = self.m
        n = self.n
        model = self.model
        num_w = int(m + m * (m + 1) / 2)

        j = 0
        t = 0
        t_step_on_loop = 0.002
        delta_idx = 5  # index jumping at t_lk

        x_list = x
        u_list = None
        w_list = None
        while True:
            Pi = None
            R = None
            while np.linalg.matrix_rank(Pi) < num_w if Pi is not None else True:  # constructing each row of matrix Theta, Xi
                for _ in range(delta_idx):
                    dv = (w.T @ self.dpi_dx(x_list[:, -1].reshape((m, 1)))).reshape((m, 1))
                    e = np.random.multivariate_normal(np.zeros(n), np.linalg.inv(self.model.R)).reshape((n, 1))
                    u = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dv + e  # (n, 1)
                    u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
                        (n, 1))  # control input after passing actuator (n, 1)
                    u = self.clipping_u(u, clipping)  # control input constraint

                    y = odeint(dyn, x_list[:, -1].reshape(m,), [t, t + t_step_on_loop], args=(u.reshape((n,)),))
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    u_list = np.hstack((u_list, u)) if u_list is not None else u
                    t = t + t_step_on_loop
                r = self.r(x_list[:, -delta_idx - 2:-1], u_list[:, -delta_idx - 1:], t_step_on_loop)
                pi = self.pi(x_list[:, -1 - delta_idx].reshape((m, 1))) - self.pi(x_list[:, -1].reshape((m, 1)))
                Pi = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                R = np.vstack((R, r)) if R is not None else r
            cond = np.linalg.cond(Pi)
            print(np.linalg.cond(Pi))
            w, _, _, _ = np.linalg.lstsq(Pi, R)
            w_list = np.hstack((w_list, w)) if w_list is not None else w
            j += 1

            if w_list.shape[1] >= 2:
                if np.linalg.norm(w_list[:, -1] - w_list[:, -2]) < tol:
                    print("Total iterations : {}".format(j))
                    break

        return w_list[:, -1].reshape((num_w, 1)), cond

    def value_iteration(self, dyn, x, w, clipping, tol):
        m = self.m
        n = self.n
        model = self.model
        num_w = int(m + m * (m + 1) / 2)

        j = 0
        t = 0
        t_step_on_loop = 0.02
        delta_idx = 3  # index jumping at t_lk

        x_list = x
        u_list = None
        w_list = w
        while True:
            Pi = None
            R = None
            W_Pi = None
            while np.linalg.matrix_rank(
                    Pi) < num_w if Pi is not None else True:  # constructing each row of matrix Theta, Xi
                for _ in range(delta_idx):
                    dv = (w.T @ self.dpi_dx(x_list[:, -1].reshape((m, 1)))).reshape((m, 1))
                    e = np.random.multivariate_normal(np.zeros(n), np.linalg.inv(self.model.R)).reshape((n, 1))
                    u = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dv + e  # (n, 1)
                    u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
                        (n, 1))  # control input after passing actuator (n, 1)
                    u = self.clipping_u(u, clipping)  # control input constraint

                    y = odeint(dyn, x_list[:, -1].reshape(m, ), [t, t + t_step_on_loop], args=(u.reshape((n,)),))
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    u_list = np.hstack((u_list, u)) if u_list is not None else u
                    t = t + t_step_on_loop
                r = self.r(x_list[:, -delta_idx - 2:-1], u_list[:, -delta_idx - 1:], t_step_on_loop)
                pi = self.pi(x_list[:, -1 - delta_idx].reshape((m, 1))) - self.pi(x_list[:, -1].reshape((m, 1)))
                Pi = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                R = np.vstack((R, r)) if R is not None else r
                w_dot_pi = (w_list[:, 0].reshape((num_w, 1)).T @ Pi(x_list[:, -1])).reshape((1, 1))
                W_Pi = np.vstack((W_Pi, w_dot_pi)) if W_Pi is not None else w_dot_pi
            cond = np.linalg.cond(Pi)
            w, _, _, _ = np.linalg.lstsq(Pi, R + W_Pi)
            w_list = np.hstack((w_list, w)) if w_list is not None else w
            j += 1

            if w_list.shape[1] >= 2:
                if np.linalg.norm(w_list[:, -1] - w_list[:, -2]) < tol:
                    print("Total iterations : {}".format(j))
                    break

        return w_list[:, -1].reshape((num_w, 1)), cond

    def sim(self, t_end, t_step, dyn, x0, x_ref, clipping=None, iteration='pi', tol='1e3'):
        m = self.m
        n = self.n
        model = self.model
        num_w = int(m + m * (m + 1) / 2)

        t = 0
        x = x0
        x_hist = x0  # (m, 1)
        u_hist = np.zeros((n, 1))  # (n, 1)
        w_hist = 0.01 * np.random.randn(num_w, 1)
        cond_list = []
        while True:
            if np.isclose(t, t_end):
                break

            if iteration == "pi":
                w, cond = self.policy_iteration(dyn, x_hist[:, -1].reshape((m, 1)), w_hist[:, -1].reshape((num_w, 1)), clipping, tol)
            elif iteration == "vi":
                w, cond = self.value_iteration(dyn, x_hist[:, -1].reshape((m, 1)), w_hist[:, -1].reshape((num_w, 1)), clipping, tol)
            cond_list.append(cond)
            dv = (w.T @ self.dpi_dx(x_hist[:, -1].reshape((m, 1)))).reshape((m, 1))
            u = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dv  # (n, 1)
            u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
                (n, 1))  # control input after passing actuator (n, 1)
            u = self.clipping_u(u, clipping)  # control input constraint

            y = odeint(dyn, x.reshape(m,), [t, t + t_step], args=(u.reshape(n,),))  # (2, m)
            x = y[-1, :].reshape((m, 1))  # (m, 1)
            x_hist = np.hstack((x_hist, x))
            u_hist = np.hstack((u_hist, u))
            w_hist = np.hstack((w_hist, w))

            t = t + t_step
            print(t)
        return x_hist, u_hist, w_hist, cond_list  # (m, time_seq), (n, time_seq)

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
