import numpy as np
from scipy.integrate import odeint

class Sim:
    def __init__(self,
                 actuator=None,
                 model=None):
        self.actuator = actuator
        self.model = model
        if model is not None:
            self.model = model
            self.m, self.n = np.shape(self.model.B)

    def actuator_u(self, u_ctrl, enabling_index=0, t_step=0.1):
        if self.actuator is not None:
            dyn = self.actuator.dynamics
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

    def policy_iteration(self, dyn, x0, w, e_shift, e_scaler, clipping, tol):
        n, m = self.n, self.m
        num_w = int(m + m * (m + 1) / 2)
        model = self.model

        j = 0
        t = 0
        t_step_on_loop = 0.0001
        delta_idx = 10
        # delta_idx = int(round(np.random.choice(range(30, 100))))  # index jumping at t_lk
        e_choice = '2'

        x_list = None
        u_list = None
        w_list = w
        while True:
            Pi = None
            R = None

            rank = 0
            rank_saturated_count = 0
            flag = True

            while np.linalg.matrix_rank(Pi) < num_w if Pi is not None else True:  # constructing each row of matrix Theta, Xi
                if x_list is not None:
                    x_list = x_list[:, -1].reshape((m, 1))
                else:
                    x_list = x0
                print('x_list{}'.format(x_list))
                # x_list = np.diag(np.random.choice([-1, 1], m)) @ np.diag(np.random.normal(1, 1, m)) @ x0
                if e_choice == '1':
                    e = 1 * np.random.multivariate_normal(np.zeros(n), np.linalg.inv(model.R)).reshape((n, 1))
                    # e = 1 * np.random.multivariate_normal(e_shift.reshape((n,)), e_scaler).reshape((n, 1))
                elif e_choice == '2':
                    a = np.array([20 * (i + 1) * np.pi * t + 0.5 * i * np.pi for i in range(n)]).reshape((n, 1))
                    e = 1 * (e_shift + e_scaler @ np.sin(a))
                for _ in range(delta_idx):
                    dv = (w_list[:, -1].reshape((num_w, 1)).T @ self.dpi_dx(x_list[:, -1].reshape((m, 1)))).reshape((m, 1))
                    u = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dv + e  # (n, 1)
                    # print("u", u)
                    y = odeint(dyn, x_list[:, -1].reshape(m,), [t, t + t_step_on_loop], args=(u.reshape((n,)),))
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    u_list = np.hstack((u_list, u)) if u_list is not None else u
                    t = t + t_step_on_loop
                r = self.r(x_list[:, -delta_idx - 2:-1], u_list[:, -delta_idx - 1:], t_step_on_loop)
                pi = self.pi(x_list[:, -1 - delta_idx].reshape((m, 1))) - self.pi(x_list[:, -1].reshape((m, 1)))
                Pi = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                R = np.vstack((R, r)) if R is not None else r

                if np.linalg.matrix_rank(Pi) == rank:
                    rank_saturated_count += 1
                if rank_saturated_count >= 5:
                    flag = False
                    print("Rank saturated, rank =", rank)
                    break
                rank = np.linalg.matrix_rank(Pi)
            cond = np.linalg.cond(Pi)
            print("PI", Pi)
            if flag:
                w, _, _, _ = np.linalg.lstsq(Pi, R)
                w_list = np.hstack((w_list, w))
                print('w{}'.format(w_list[:, -1]))
                j += 1

            if w_list.shape[1] >= 3:
                if np.linalg.norm(w_list[:, -2] - w_list[:, -1]) < tol:
                    print('Converged in', j, 'iteration')
                    print('w{}'.format(w_list[:, -1]))
                    break

        return w_list[:, -1].reshape((num_w, 1)), cond

    def value_iteration(self, dyn, x, w, e_shift, e_scaler, clipping, tol):
        n, m = self.n, self.m
        num_w = int(m + m * (m + 1) / 2)
        model = self.model

        j = 0
        t = 0
        t_step_on_loop = 0.001
        delta_idx = 10
        # delta_idx = int(round(np.random.choice(range(30, 100))))  # index jumping at t_lk
        e_choice = '1'

        x_list = None
        u_list = None
        w_list = w
        while True:
            Pi = None
            R = None
            W_Pi = None

            rank = 0
            rank_saturated_count = 0
            flag = True

            while np.linalg.matrix_rank(Pi) < num_w if Pi is not None else True:  # constructing each row of matrix Theta, Xi
                if x_list is not None:
                    x_list = x_list[:, -1].reshape((m, 1))
                else:
                    x_list = x
                # x_list = np.diag(np.random.choice([-1, 1], m)) @ np.diag(np.random.normal(1, 1, m)) @ x
                if e_choice == '1':
                    e = 1 * np.random.multivariate_normal(np.zeros(n), np.linalg.inv(model.R)).reshape((n, 1))
                    # e = 0.1 * np.random.multivariate_normal(e_shift.reshape((n,)), e_scaler).reshape((n, 1))
                    e = 0
                elif e_choice == '2':
                    a = np.array([20 * (i + 1) * np.pi * t + 0.5 * i * np.pi for i in range(n)]).reshape((n, 1))
                    e = 1 * e_shift + e_scaler @ np.sin(a)
                    e = 0
                for _ in range(delta_idx):
                    dv = (w.T @ self.dpi_dx(x_list[:, -1].reshape((m, 1)))).reshape((m, 1))
                    u = -0.5 * np.linalg.inv(model.R) @ model.B.T @ dv + e  # (n, 1)

                    y = odeint(dyn, x_list[:, -1].reshape(m, ), [t, t + t_step_on_loop], args=(u.reshape((n,)),))
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    u_list = np.hstack((u_list, u)) if u_list is not None else u
                    t = t + t_step_on_loop
                r = self.r(x_list[:, -delta_idx - 2:-1], u_list[:, -delta_idx - 1:], t_step_on_loop)
                pi = self.pi(x_list[:, -1 - delta_idx].reshape((m, 1))) - self.pi(x_list[:, -1].reshape((m, 1)))
                Pi = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                R = np.vstack((R, r)) if R is not None else r
                w_dot_pi = (w_list[:, -1].reshape((num_w, 1)).T @ self.pi(x_list[:, -1].reshape((m, 1)))).reshape((1, 1))
                W_Pi = np.vstack((W_Pi, w_dot_pi)) if W_Pi is not None else w_dot_pi

                if np.linalg.matrix_rank(Pi) == rank:
                    rank_saturated_count += 1
                if rank_saturated_count >= 5:
                    flag = False
                    print("Rank saturated")
                    break
                rank = np.linalg.matrix_rank(Pi)
            cond = np.linalg.cond(Pi)

            if flag:
                w, _, _, _ = np.linalg.lstsq(Pi, R + W_Pi)
                w_list = np.hstack((w_list, w))
                print(w)
                j += 1

            if w_list.shape[1] >= 2:
                if np.linalg.norm(w_list[:, -2] - w_list[:, -1]) < tol:
                    print('Converged in', j, 'iteration')
                    print('w{}'.format(w_list[:, -1]))
                    break

        return w_list[:, -1].reshape((num_w, 1)), cond

    def sim(self, t_end, t_step, dyn, x0, x_ref, e_shift, e_scaler, clipping=None, iteration='pi', tol='1e3'):
        n, m = self.n, self.m
        model = self.model
        num_w = int(m + m * (m + 1) / 2)

        t = 0
        x = x0
        x_hist = x0  # (m, 1)
        u_hist = np.zeros((n, 1))  # (n, 1)
        w_hist = 0.00 * np.random.randn(num_w, 1)
        cond_list = []
        w_fixed = False

        while True:
            if np.isclose(t, t_end):
                break
            if w_fixed is False:
                if iteration == "pi":
                    w, cond = self.policy_iteration(dyn, x_hist[:, -1].reshape((m, 1)),
                                                         w_hist[:, -1].reshape((num_w, 1)),
                                                         e_shift, e_scaler, clipping, tol)
                elif iteration == "vi":
                    w, cond = self.value_iteration(dyn, x_hist[:, -1].reshape((m, 1)),
                                                        w_hist[:, -1].reshape((num_w, 1)),
                                                        e_shift, e_scaler, clipping, tol)
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

            if w_fixed is False and np.linalg.norm(w_hist[:, -2] - w_hist[:, -1]) < tol:
                print('w fixed at t=', t)
                w_fixed = True

            t = t + t_step
        return x_hist, u_hist, w_hist, cond_list  # (m, time_seq), (n, time_seq)