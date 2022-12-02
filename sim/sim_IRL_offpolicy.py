import numpy as np
from scipy.integrate import odeint
from control import lqr

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

    def iteration(self, x0, clipping, dyn, u0_shift, u0_scaler, tol):
        n = self.n
        m = self.m
        model = self.model
        P_list = []  # P_0, P_1, ... len of k
        P_compute = []
        K_list = []  # K_0, K_1, ... len of k+1
        cond_list = []
        P = np.zeros((m, m))
        P_list.append(P)
        K = 0.001 * np.random.randn(n, m)  # Initial gain matrix
        K_list.append(K)
        k = 0

        x_list = None  # (m, 1)
        u0_list = None

        while True:
            Q = model.Q + K_list[-1].T @ model.R @ K_list[-1]  # Q_k

            rank = 0
            t_lk = 0
            t_step_on_loop = 0.0001
            delta_idx = int(round(np.random.choice(range(30, 100))))  # index jumping at t_lk
            print("k = {}".format(k))
            Theta = None
            Xi = None
            rank_saturated_count = 0
            flag = True
            theta_xx = None
            theta_xu = None
            delta_xx = None
            u0_choice = '1'

            # while np.linalg.matrix_rank(np.hstack([theta_xx, theta_xu])) < m * (m + 1) / 2 + m * n or np.linalg.cond(Theta) > 1e2 if Theta is not None else True:  # constructing each row of matrix Theta, Xi
            while np.linalg.matrix_rank(np.hstack([theta_xx, theta_xu])) < m * (m + 1) / 2 + m * n if Theta is not None else True:  # constructing each row of matrix Theta, Xi
                # x_list = np.hstack((x_list, np.random.randn(m,1)))
                # x_list = np.hstack((x_list, np.random.multivariate_normal(np.zeros(m), np.diag(np.abs(x0).squeeze())).reshape((m, 1))))
                x_list = np.hstack((x_list, np.diag(np.random.choice([-1, 1], m)) @ np.diag(np.random.normal(1, 1, m)) @ x0)) if x_list is not None else np.diag(np.random.choice([-1, 1], m)) @ np.diag(np.random.normal(1, 1, m)) @ x0
                if u0_choice == '1':
                    u0_list = np.hstack((u0_list, np.random.multivariate_normal(u0_shift.reshape((n,)), u0_scaler).reshape((n, 1)))) if u0_list is not None else np.random.multivariate_normal(u0_shift.reshape((n,)), u0_scaler).reshape((n, 1))
                elif u0_choice == '2':
                    a = np.array([20 * (i + 1) * np.pi * t_lk + 0.5 * i * np.pi for i in range(n)]).reshape((n, 1))
                    u0_list = np.hstack((u0_list, u0_shift + u0_scaler @ np.sin(a))) if u0_list is not None else u0_shift + u0_scaler @ np.sin(a)
                fx1_list = np.kron(x_list[:, -1].T, x_list[:, -1].T).reshape((1, m*m))  # (1, mm) # used for integral of theta_xx
                fx2_list = np.kron(x_list[:, -1].T, u0_list[:, -1].T).reshape((1, m*n))  # (1, 1) # used for integral of theta_xu
                for _ in range(delta_idx):  # delta_idx element constructs one row of Theta matrix
                    u = u0_list[:, -1].reshape((n, 1))
                    # print(u)
                    y = odeint(dyn, x_list[:, -1].reshape((m,)), [t_lk, t_lk + t_step_on_loop],
                               args=(u.reshape(n, ),))  # size of (2, n)

                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    fx1_list = np.vstack((fx1_list, np.kron(x_list[:, -1].T, x_list[:, -1].T).reshape((1, m*m))))  # size of (delta_idx+1, mn) after loop  # used for integral of theta_xx
                    fx2_list = np.vstack((fx2_list, np.kron(x_list[:, -1].T, u.T).reshape((1, m*n))))  # size of (delta_idx+1, 1) after loop   # used for integral of theta_xu
                    t_lk = t_lk + t_step_on_loop

                fx1_integral = np.trapz(fx1_list, dx=t_step_on_loop, axis=0).reshape((1, m*m))
                fx2_integral = np.trapz(fx2_list, dx=t_step_on_loop, axis=0).reshape((1, m*n))
                theta_xx = np.vstack((theta_xx, fx1_integral)) if theta_xx is not None else fx1_integral
                theta_xu = np.vstack((theta_xu, fx2_integral)) if theta_xu is not None else fx2_integral
                delta_xx = np.vstack((delta_xx, (np.kron(x_list[:, -1].T, x_list[:, -1].T) - np.kron(x_list[:, -delta_idx - 1].T, x_list[:, -delta_idx - 1].T)).reshape((1, m * m)))) if delta_xx is not None else (np.kron(x_list[:, -1].T, x_list[:, -1].T) - np.kron(x_list[:, -delta_idx - 1].T, x_list[:, -delta_idx - 1].T)).reshape((1, m * m)) # size of (lines, mm)
                element_1 = -2 * theta_xx @ np.kron(np.eye(m), K_list[-1].T @ self.model.R) - 2 * theta_xu @ np.kron(np.eye(m), self.model.R)
                Theta = np.hstack((delta_xx, element_1))  # size of (rows, nn+ mn)
                Xi = -theta_xx @ Q.reshape((m*m, 1))    # size of (rows, 1)

                if np.linalg.matrix_rank(np.hstack([theta_xx, theta_xu])) == rank:
                    rank_saturated_count += 1
                if rank_saturated_count >= 5:
                    flag = False
                    print("Rank saturated")
                    break
                rank = np.linalg.matrix_rank(Theta)
            cond_list.append(np.linalg.cond(np.hstack([theta_xx, theta_xu])))
            # # Making symmetric matrix P, and gain matrix K
            # if flag:
            #     idx = np.where(np.tril(np.full((m, m), 1), -1).reshape((m*m), order="F") == 1)[0]
            #     mask = np.ones((m*m + m*n,), dtype=bool)
            #     mask[idx] = False
            #     Theta_aug = Theta[:, mask]
            #     sol, _, _, _ = np.linalg.lstsq(Theta_aug, Xi)  # size of (mm + mn, 1)
            #     P = np.zeros((m*m,))
            #     P[mask[:-m*n]] = sol[:int(m * (m + 1) / 2)].squeeze()
            #     P = P.reshape((m, m), order='F')    # upper triangular matrix
            #     P = P + np.triu(P, 1).T
            #     P_compute.append(P)
            #     P_list.append(0.1 * P + 0.9 * P_list[-1])
            #     # P_list.append(P)
            #     # print(P)
            #     K = sol[int(m * (m + 1) / 2):].reshape((n, m), order='F')
            #     # print(K)
            #     K_list.append(0.1 * K + 0.9 * K_list[-1])
            #     # K_list.append(K)
            #     if np.max(abs(P_list[-1])) > constraint_P or np.max(abs(K_list[-1])) > constraint_K and len(K_list) >= 2:  # Ignore some bad cases
            #         print("Ignoring overly diverging P, K solutions")
            #         del P_list[-1]
            #         del K_list[-1]
            #         continue
            #     k += 1

            if flag:
                sol, _, _, _ = np.linalg.lstsq(Theta, Xi)  # size of (mm + mn, 1)
                P = sol[:m*m].reshape((m, m))
                P_compute.append(P)
                P_list.append(P)
                # P_list.append(0.01 * P + 0.99 * P_list[-1])
                K = sol[m*m:].reshape((n, m), order='F')
                # print(K)
                # K_list.append(0.01 * K + 0.99 * K_list[-1])
                K_list.append(K)
                # if np.max(abs(P_list[-1])) > constraint_P or np.max(abs(K_list[-1])) > constraint_K and len(K_list) >= 2:  # Ignore some bad cases
                #     print("Ignoring overly diverging P, K solutions")
                #     del P_list[-1]
                #     del K_list[-1]
                #     continue
                print(P)
                k += 1

            if len(P_compute) >= 10:
                P_avg = np.mean(np.stack(P_compute[-10:], axis=0), axis=0)
                print(np.linalg.norm(np.max(np.stack(P_compute[-10:], axis=0)) - np.min(np.stack(P_compute[-10:], axis=0))))
                if np.linalg.norm(np.max(np.stack(P_compute[-10:], axis=0)) - np.min(np.stack(P_compute[-10:], axis=0))) < tol:
                    print("Total iterations : {}".format(k))
                    print("P : {}".format(P))
                    print("K : {}".format(K))
                    break
        return P_list, K_list, cond_list

    def sim(self, t_end, t_step, dyn, x0, x_ref, u0_shift, u0_scaler, clipping=None, tol=1e-3):
        m = self.m
        n = self.n
        model = self.model
        K_opt, _, _ = lqr(model.A, model.B, model.Q, model.R)

        P_list, K_list, cond_list = self.iteration(x0, clipping, dyn, u0_scaler, u0_shift, tol)
        t = 0
        x = x0

        K = K_list[-1]
        print(np.linalg.norm(K - K_opt) / np.linalg.norm(K_opt) * 1e2, "% difference with optimal solution)")
        x_hist = x0  # (m, 1)
        u_hist = -K @ (x - x_ref)  # (n, 1)
        while True:
            if np.isclose(t, t_end):
                break

            u = -K @ (x - x_ref)  # (n, 1)
            u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
                (n, 1))  # control input after passing actuator (n, 1)
            u = self.clipping_u(u, clipping)  # control input constraint

            y = odeint(dyn, x.reshape(m, ), [t, t + t_step], args=(u.reshape(n, ),))  # (2, m)
            x = y[-1, :].reshape((m, 1))  # (m, 1)
            x_hist = np.hstack((x_hist, x))
            u_hist = np.hstack((u_hist, u))

            t = t + t_step
        return x_hist, u_hist, P_list, K_list, cond_list  # (m, time_seq), (n, time_seq)