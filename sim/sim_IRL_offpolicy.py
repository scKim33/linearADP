import numpy as np
from scipy.integrate import odeint
from model.actuator import Actuator
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

    def iteration(self, x0, clipping, dyn):
        n = self.n
        m = self.m
        model = self.model
        P_list = []  # P_0, P_1, ... len of k
        K_list = []  # K_0, K_1, ... len of k+1
        K = 0.01 * np.random.randn(n, m)  # Initial gain matrix
        K_list.append(K)
        x = x0

        while True:
            Q = model.Q + K_list[-1].T @ model.R @ K_list[-1]

            line = 0  # number of row line of Theta_k
            t_lk = 0
            t_step_on_loop = 0.02
            delta_idx = 5  # index jumping at t_lk
            Theta = None
            Xi = None
            theta_xx = None
            theta_xu = None
            x_list = x0  # (m, 1)
            u0 = np.random.randn(n, 1)
            while np.linalg.matrix_rank(np.hstack([theta_xx, theta_xu])) < n * (n + 1) / 2 + m * n and np.linalg.cond(
                    Theta) > 1e2 if Theta is not None else True:  # constructing each row of matrix Theta, Xi
                fx1_list = np.kron(x_list[:, -1].T, x_list[:, -1].T).reshape((1, m*m))  # (1, mm) # used for integral of theta_xx
                fx2_list = np.kron(x_list[:, -1].T, u0.T).reshape((1, m*n))  # (1, 1) # used for integral of theta_xu
                for _ in range(delta_idx):  # delta_idx element constructs one row of Theta matrix
                    u = u0
                    u = self.actuator_u(u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
                        (n, 1))  # control input after passing actuator (n, 1)
                    u = self.clipping_u(u, clipping)  # control input constraint

                    y = odeint(dyn, x_list[:, -1].reshape((m,)), [t_lk, t_lk + t_step_on_loop],
                               args=(u.reshape(n, ),))  # size of (2, n)
                    x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
                    fx1_list = np.vstack((fx1_list, np.kron(x_list[:, -1].T, x_list[:, -1].T).reshape((1, m*m))))  # size of (delta_idx+1, mn) after loop  # used for integral of theta_xx
                    fx2_list = np.vstack((fx2_list, np.kron(x_list[:, -1].T, u0.T).reshape((1, m*n))))  # size of (delta_idx+1, 1) after loop   # used for integral of theta_xu
                    t_lk = t_lk + t_step_on_loop

                fx1_integral = np.trapz(fx1_list, dx=t_step_on_loop, axis=0).reshape((1, m*m))
                fx2_integral = np.trapz(fx2_list, dx=t_step_on_loop, axis=0).reshape((1, m*n))
                theta_xx = np.vstack((theta_xx, fx1_integral)) if theta_xx is not None else fx1_integral
                theta_xu = np.vstack((theta_xu, fx2_integral)) if theta_xu is not None else fx2_integral
                delta_xx = (np.kron(x_list[:, -delta_idx - 1].T, x_list[:, -delta_idx - 1].T) - np.kron(x_list[:, -1].T, x_list[:, -1].T)).reshape((1, m * m))  # size of (1, mm)
                element_1 = -2 * theta_xx @ np.kron(np.eye(m), K_list[-1].T @ self.model.R) -2 * theta_xu @ np.kron(np.eye(m), self.model.R)
                Theta = np.vstack((Theta, np.hstack((delta_xx, element_1)))) if Theta is not None else np.hstack(
                    (delta_xx, element_1))  # size of (rows, nn+ mn)
                Xi = np.vstack((Xi, -theta_xx @ Q.reshape((m*m, 1)))) if Xi is not None else -theta_xx @ Q.reshape((m*m, 1))  # size of (rows, 1)

            sol, _, _, _ = np.linalg.lstsq(Theta, Xi)  # size of (mm + mn, 1)
            sol.reshape((m * m + m * n, 1))
            P = sol[:m * m].reshape((m, m))
            P_list.append(P)
            K = sol[m * m:].reshape((n, m), order='F')
            K_list.append(K)

            if len(P_list) > 2:
                if np.linalg.norm(P_list[-1] - P_list[-2]) < 1e-2:
                    break
        return P_list, K_list

    def sim(self, t_end, t_step, dyn, x0, x_ref, clipping=None):
        m = self.m
        n = self.n
        model = self.model

        P_list, K_list = self.iteration(x0, clipping, dyn)
        t = 0
        x = x0
        # K, _, _ = lqr(model.A, model.B, model.Q, model.R) # This is standard LQR result
        K = K_list[-1]  # This is off policy result
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
        return x_hist, u_hist  # (m, time_seq), (n, time_seq)

#
# def actuator_u(actuator, u_ctrl, enabling_index=0, t_step=0.1):
#     """
#     :param t_step: input signal time delay by the actuator
#     :param actuator: Actuator model
#     :param u_ctrl: Control input entering actuator from controller
#     :param enabling_index: Pre-determined input element which is affected by actuator
#     :return:
#     """
#     u_act = np.array([u_ctrl[enabling_index], 0])  # u_act, u_act_dot in systems of ODE
#     u_act = odeint(actuator.dynamics, u_act, [0, t_step],
#                    args=(u_ctrl[enabling_index],))  # only considering enabling_index element
#     u_act = u_act[-1, :]  # take u_act at t_step
#     u_ctrl[0] = u_act[0]
#     return u_ctrl   # same size as input u_ctrl
#
#
# def clipping_u(model, u, clipping):
#     '''
#     :param model:
#     :param u:
#     :param clipping:
#     :return:
#     '''
#     if clipping is not None:
#         n = np.shape(model.B)[1]
#         u = u.reshape((n, 1))  # Reshaping u of dim=2
#         for u_i, constraint, i in zip(u, clipping, range(n)):
#             u[i, 0] = np.clip(u_i, constraint[0], constraint[1])  # constraint of u
#     return u # (n, 1)
#
# def on_policy_iteration(x0, m, n, model, actuator, clipping, dyn):
#     P_list = []  # P_0, P_1, ... len of k
#     K_list = []  # K_0, K_1, ... len of k+1
#     x = x0
#
#     while True:
#         K = np.zeros((n, m))  # Initial gain matrix : zero matrix
#         K_list.append(K)
#         Q = model.Q + K_list[-1].T @ model.R @ K_list[-1]
#
#         line = 0  # number of row line of Theta_k
#         t_lk = 0
#         t_step_onloop = 0.02
#         delta_idx = 5  # index jumping at t_lk
#         Theta = None
#         Xi = None
#         x_list = x0  # (m, 1)
#         e_list = np.random.randn(n, 1)
#         while np.linalg.matrix_rank(Theta) < n * (n + 1) / 2 + m * n and np.linalg.cond(
#                 Theta) > 1e3 if Theta is not None else True:  # constructing each row of matrix Theta, Xi
#             fx1_list = np.kron(x_list[:, -1],
#                                e_list[:, -1].T @ model.R)  # (1, mn) # used for integral of Theta, Xi matrix
#             fx2_list = (-x_list[:, -1].T @ Q @ x_list[:, -1]).reshape((1, 1))  # (1, 1)
#             for _ in range(delta_idx):  # delta_idx element constructs one row of Theta matrix
#                 e = np.random.randn(n, 1)
#                 u = -K_list[-1] @ x_list[:, -1].reshape((m, 1)) + e  # (n, 1)
#                 u = actuator_u(actuator, u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
#                     (n, 1))  # control input after passing actuator (n, 1)
#                 u = clipping_u(model, u, clipping)  # control input constraint
#
#                 y = odeint(dyn, x_list[:, -1].reshape((m,)), [t_lk, t_lk + t_step_onloop], args=(u.reshape(n, ),))  # size of (2, n)
#                 x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
#                 e_list = np.hstack((e_list, e))
#                 fx1_list = np.vstack((fx1_list, np.kron(x_list[:, -1].T, e_list[:,
#                                                                          -1].T @ model.R)))  # size of (delta_idx+1, mn) after loop  # used for integral of Theta, Xi matrix
#                 fx2_list = np.vstack((fx2_list, (-x_list[:, -1].T @ Q @ x_list[:, -1]).reshape(
#                     (1, 1))))  # size of (delta_idx+1, 1) after loop
#                 t_lk = t_lk + t_step_onloop
#
#             element_1 = (np.kron(x_list[:, -delta_idx - 1].T, x_list[:, -delta_idx - 1].T) - np.kron(x_list[:, -1].T,
#                                                                                                      x_list[:,
#                                                                                                      -1].T)).reshape(
#                 (1, m * m))  # size of (1, mm)
#             element_2 = (-2 * np.trapz(fx1_list, dx=t_step_onloop, axis=0)).reshape(
#                 (1, m * n))  # size fx_list : (delta_idx+1, mn)
#             element_3 = np.trapz(fx2_list, dx=t_step_onloop, axis=0)
#             Theta = np.vstack((Theta, np.hstack((element_1, element_2)))) if Theta is not None else np.hstack(
#                 (element_1, element_2))  # size of (rows, nn+ mn)
#             Xi = np.vstack((Xi, np.array([element_3]).reshape((1, 1)))) if Xi is not None else np.array(
#                 [element_3]).reshape((1, 1))  # size of (rows, 1)
#
#         sol, _, _, _ = np.linalg.lstsq(Theta, Xi)  # size of (mm + mn, 1)
#         sol.reshape((m * m + m * n, 1))
#         P = sol[:m * m].reshape((m, m))
#         P_list.append(P)
#         K = sol[m * m:].reshape((n, m), order='F')
#         K_list.append(K)
#
#         if len(P_list) > 2:
#             if np.linalg.norm(P_list[-1] - P_list[-2]) < 1e-2:
#                 break
#     return P_list, K_list
#
# def sim_IRL_onpolicy(t_end, t_step, model, actuator, dyn, x0, x_ref, clipping=None, method=None,
#             actuator_status=False):
#     """
#     :param t_end:
#     :param t_step:
#     :param model:
#     :param actuator:
#     :param dyn:
#     :param x0:
#     :param x_ref:
#     :param clipping:
#     :param method:
#     :param actuator_status:
#     :return:
#     """
#
#     # x = x0
#     m = np.shape(model.A)[1]  # dimension of x
#     n = np.shape(model.B)[1]  # dimension of u
#     # P_list = []  # P_0, P_1, ... len of k
#     # K_list = []  # K_0, K_1, ... len of k+1
#     #
#     # while True:
#     #     K = np.zeros((n, m))  # Initial gain matrix : zero matrix
#     #     K_list.append(K)
#     #     Q = model.Q + K_list[-1].T @ model.R @ K_list[-1]
#     #
#     #     line = 0    # number of row line of Theta_k
#     #     t_lk = 0
#     #     t_step_onloop = 0.02
#     #     delta_idx = 5   # index jumping at t_lk
#     #     Theta = None
#     #     Xi = None
#     #     x_list = x0 # (m, 1)
#     #     e_list = np.random.randn(n, 1)
#     #     while np.linalg.matrix_rank(Theta) < n * (n + 1) / 2 + m * n and np.linalg.cond(Theta) > 1e3 if Theta is not None else True:  # constructing each row of matrix Theta, Xi
#     #         fx1_list = np.kron(x_list[:, -1], e_list[:, -1].T @ model.R)  # (1, mn) # used for integral of Theta, Xi matrix
#     #         fx2_list = (-x_list[:, -1].T @ Q @ x_list[:, -1]).reshape((1, 1))  # (1, 1)
#     #         for _ in range(delta_idx):  # delta_idx element constructs one row of Theta matrix
#     #             e = np.random.randn(n, 1)
#     #             u = -K_list[-1] @ x_list[:, -1].reshape((m, 1)) + e  # (n, 1)
#     #             u = actuator_u(actuator, u.reshape((n,)), enabling_index=0, t_step=0.1).reshape((n, 1)) # control input after passing actuator (n, 1)
#     #             u = clipping_u(model, u, clipping)  # control input constraint
#     #
#     #             y = odeint(dyn, x.reshape((m,)), [t_lk, t_lk + t_step_onloop], args=(u.reshape(n,),))    # size of (2, n)
#     #             x_list = np.hstack((x_list, y[-1, :].reshape((m, 1))))
#     #             e_list = np.hstack((e_list, e))
#     #             fx1_list = np.vstack((fx1_list, np.kron(x_list[:, -1].T, e_list[:, -1].T @ model.R))) # size of (delta_idx+1, mn) after loop  # used for integral of Theta, Xi matrix
#     #             fx2_list = np.vstack((fx2_list, (-x_list[:, -1].T @ Q @ x_list[:, -1]).reshape((1, 1)))) # size of (delta_idx+1, 1) after loop
#     #             t_lk = t_lk + t_step_onloop
#     #
#     #         element_1 = (np.kron(x_list[:, -delta_idx-1].T, x_list[:, -delta_idx-1].T) - np.kron(x_list[:, -1].T, x_list[:, -1].T)).reshape((1, m*m))   # size of (1, mm)
#     #         element_2 = (-2 * np.trapz(fx1_list, dx=t_step_onloop, axis=0)).reshape((1, m*n))    # size fx_list : (delta_idx+1, mn)
#     #         element_3 = np.trapz(fx2_list, dx=t_step_onloop, axis=0)
#     #         Theta = np.vstack((Theta, np.hstack((element_1, element_2)))) if Theta is not None else np.hstack((element_1, element_2))   # size of (rows, nn+ mn)
#     #         Xi = np.vstack((Xi, np.array([element_3]).reshape((1, 1)))) if Xi is not None else np.array([element_3]).reshape((1, 1)) # size of (rows, 1)
#     #
#     #     sol, _, _, _ = np.linalg.lstsq(Theta, Xi)  # size of (mm + mn, 1)
#     #     sol.reshape((m*m + m*n, 1))
#     #     P = sol[:m*m].reshape((m, m))
#     #     P_list.append(P)
#     #     K = sol[m*m:].reshape((n, m), order='F')
#     #     K_list.append(K)
#     #
#     #     if len(P_list) > 2:
#     #         if np.linalg.norm(P_list[-1] - P_list[-2]) < 1e-2:
#     #             break
#
#     P_list, K_list = on_policy_iteration(x0, np.shape(model.A)[1], np.shape(model.B)[1], model, actuator, clipping, dyn)
#     t = 0
#     x = x0
#     # K, _, _ = lqr(model.A, model.B, model.Q, model.R) # This is standard LQR result
#     K = - np.linalg.inv(model.R) @ model.B.T @ P_list[-1]   # This is on policy result
#     x_hist = x0 # (m, 1)
#     u_hist = -K @ (x - x_ref)   # (n, 1)
#     while True:
#         if np.isclose(t, t_end):
#             break
#
#         u = -K @ (x - x_ref)   # (n, 1)
#         u = actuator_u(actuator, u.reshape((n,)), enabling_index=0, t_step=0.1).reshape(
#             (n, 1))  # control input after passing actuator (n, 1)
#         u = clipping_u(model, u, clipping)  # control input constraint
#
#         y = odeint(dyn, x.reshape(m,), [t, t + t_step], args=(u.reshape(n,),)) # (2, m)
#         x = y[-1, :].reshape((m, 1))  # (m, 1)
#         x_hist = np.hstack((x_hist, x))
#         u_hist = np.hstack((u_hist, u))
#
#         t = t + t_step
#     return x_hist, u_hist   # (m, time_seq), (n, time_seq)