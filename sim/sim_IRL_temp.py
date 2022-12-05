import numpy as np
from scipy.integrate import odeint
from control import lqr


class Sim:
    def __init__(self,
                 actuator=None,
                 model=None):
        self.actuator = actuator
        self.model = model
        if model is not None:
            self.model = model
            self.m, self.n = np.shape(self.model.B)
        self.Rinv = np.linalg.inv(model.R)

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
        num_w = int(m * (m + 1) / 2)
        pi = np.zeros((num_w, 1))
        # for i in range(m):
        #     pi[i, 0] = x[i]
        el = 0
        for j in range(m):
            for k in range(j, m):
                pi[el, 0] = x[j] * x[k]
                el += 1
        return pi

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

    def iteration(self, dyn, x0, w, iteration, e_shift, e_scaler, clipping, tol):
        n, m = self.n, self.m
        num_w = int(m * (m + 1) / 2)
        model = self.model

        j = 0
        t = 0
        t_step_on_loop = 0.001
        delta_idx = 40
        # delta_idx = int(round(np.random.choice(range(30, 100))))  # index jumping at t_lk
        e_choice = '1'

        x_list = None
        u_list = None
        w_list = w
        cond_list = []
        while True:
            Pi = None
            R = None
            # W_Pi = None

            rank = 0
            rank_saturated_count = 0
            flag = True

            while np.linalg.matrix_rank(Pi) < num_w - 4 if Pi is not None else True:  # constructing each row of matrix Theta, Xi
                if x_list is not None:
                    x_list = x_list[:, -1].reshape((m, 1))
                else:
                    x_list = x0
                for _ in range(delta_idx):
                    if e_choice == '1':
                        e = 0 * np.random.multivariate_normal(np.zeros(n), np.linalg.inv(model.R)).reshape((n, 1))
                        # e = 1 * np.random.multivariate_normal(e_shift.reshape((n,)), e_scaler).reshape((n, 1))
                    elif e_choice == '2':
                        a = np.array([20 * (i + 1) * np.pi * t + 0.5 * i * np.pi for i in range(n)]).reshape((n, 1))
                        e = 1 * (e_shift + e_scaler @ np.sin(a))
                    _w = w_list[:, -1]
                    P = np.array([[_w[0], _w[1]/2, _w[2]/2, _w[3]/2],
                                  [_w[1]/2, _w[4], _w[5]/2, _w[6]/2],
                                  [_w[2]/2, _w[5]/2, _w[7], _w[8]/2],
                                  [_w[3]/2, _w[6]/2, _w[8]/2, _w[9]]])
                    K = self.Rinv @ model.B.T @ P
                    u = - K @ x_list[:, -1].reshape((m, 1)) + e  # (n, 1)
                    y = odeint(dyn, x_list[:, -1].reshape(m,), [t, t + t_step_on_loop], args=(u.reshape((n,)),))
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    u_list = np.hstack((u_list, u)) if u_list is not None else u
                    t = t + t_step_on_loop

                # r = self.r(x_list[:, -delta_idx - 2:-1], u_list[:, -delta_idx - 1:], t_step_on_loop)
                # pi = self.pi(x_list[:, -1 - delta_idx].reshape((m, 1)))
                r = self.r(x_list[:, :delta_idx], u_list[:, :], t_step_on_loop)
                pi = self.pi(x_list[:, 1].reshape((m, 1)))
                Pi_temp = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                if np.linalg.matrix_rank(Pi_temp) > rank:
                    print(np.linalg.matrix_rank(Pi_temp))
                    # Pi = np.vstack((Pi, pi.T)) if Pi is not None else pi.T
                    Pi = Pi_temp
                    if R is not None:
                        R = np.vstack((R, r + w_list[:, -1] @ self.pi(x_list[:, -1].reshape((m, 1)))))
                    else:
                        R = r + w_list[:, -1] @ self.pi(x_list[:, -1].reshape((m, 1)))

                if np.linalg.matrix_rank(Pi) == rank:
                    rank_saturated_count += 1
                if rank_saturated_count >= 5:
                    flag = False
                    print("Rank saturated, rank =", rank)
                    break
                rank = np.linalg.matrix_rank(Pi)
                # print(rank)
            cond = np.linalg.cond(Pi)
            cond_list.append(cond)

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
            else:
                break

        return w_list, cond_list

    def sim(self, t_end, t_step, dyn, x0, x_ref, e_shift, e_scaler, clipping=None, iteration='pi', tol='1e3'):
        n, m = self.n, self.m
        num_w = int(m * (m + 1) / 2)
        model = self.model
        K_opt, _, _ = lqr(model.A, model.B, model.Q, model.R)

        w0 = 0.00 * np.random.randn(num_w, 1)  # Initial gain matrix
        w_list, cond_list = self.iteration(dyn, x0, w0, iteration, e_shift, e_scaler, clipping, tol)
        _w = w_list[:, -1]
        P = np.array([[_w[0], _w[1]/2, _w[2]/2, _w[3]/2],
                      [_w[1]/2, _w[4], _w[5]/2, _w[6]/2],
                      [_w[2]/2, _w[5]/2, _w[7], _w[8]/2],
                      [_w[3]/2, _w[6]/2, _w[8]/2, _w[9]]])
        t = 0
        x = x0
        K = self.Rinv @ model.B.T @ P
        print(np.linalg.norm(K - K_opt) / np.linalg.norm(K_opt) * 1e2, "% difference with optimal solution)")
        x_hist = x0  # (m, 1)
        u_hist = -K @ (x - x_ref)  # (n, 1)
        while True:
            if np.isclose(t, t_end):
                break

            u = -K @ (x - x_ref)  # (n, 1)
            # u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
            #     (n, 1))  # control input after passing actuator (n, 1)
            # u = self.clipping_u(u, clipping)  # control input constraint

            y = odeint(dyn, x.reshape(m,), [t, t + t_step], args=(u.reshape(n,),))  # (2, m)
            x = y[-1, :].reshape((m, 1))  # (m, 1)
            x_hist = np.hstack((x_hist, x))
            u_hist = np.hstack((u_hist, u))

            t = t + t_step
        return x_hist, u_hist, w_list, cond_list  # (m, time_seq), (n, time_seq)
