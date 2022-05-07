from simple_pid import PID
import numpy as np
from scipy.integrate import odeint

def Sim(t_end, t_step, model, x0, args, controller):
    """
    Linear model simulation
    :param model: State-space form linear model
    :param args: Tuple; coefficient used at the model, careful to the input sequence
    :param controller: Two kinds of controller ; "PID" or "LQR"
    :param x0: Initial condition of the system
    :param t_end: Time at which simulation terminates
    :param t_step: Time step of the simulation
    :return: System variable x (vector form)
    """

    t = np.linspace(0, t_end, int(t_end / t_step) + 1)
    x = np.zeros((len(x0), len(t)))
    u = np.zeros(len(t))
    if controller == "PID":
        for i in range(len(t) - 1):
            u[i + 1] = pid(x[0, i], dt=t_step)
            # simulation condition -> set dt equal to simulation time step
            # if not, pid takes value as real time step
            ts = [t[i], t[i + 1]]
            y = odeint(model, x0, ts, args=((u[i + 1],) + args))
            x[:, i + 1] = y[-1, :]
            x0 = x[:, i + 1]
    elif controller == "LQR":
        pass
    return x, u