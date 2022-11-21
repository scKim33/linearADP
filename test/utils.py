import matplotlib.pyplot as plt
import numpy as np


def plot(x_hist, u_hist, tspan, x_ref, u_ref, type='plot', x_shape=None, u_shape=None, x_label=None, u_label=None):
    m = x_hist.shape[0]
    n = u_hist.shape[0]

    plt.figure()
    for i in range(x_shape[0] * x_shape[1]):
        plt.subplot(x_shape[0], x_shape[1], i + 1)
        if type == 'plot':
            plt.plot(tspan, x_hist[i, :], 'k-', linewidth=1.2)
        elif type == 'scatter':
            plt.scatter(tspan, x_hist[i, :], s=15, c='k', marker='x', linewidth=1.2)
        plt.plot(tspan, x_ref[i] * np.ones(len(tspan)), 'r--', linewidth=1.2)
        plt.grid()
        plt.xlim([tspan[0], tspan[-1]])
        plt.ylabel(r'{}'.format(x_label[i]))
        plt.title('State trajectory')
        plt.legend(('State', 'Reference'))

    plt.figure()
    for i in range(u_shape[0] * u_shape[1]):
        plt.subplot(u_shape[0], u_shape[1], i + 1)
        if type == 'plot':
            plt.plot(tspan, u_hist[i, :], 'b-', linewidth=1.2)
        elif type == 'scatter':
            plt.scatter(tspan, u_hist[i, :], s=15, c='b', marker='x', linewidth=1.2)
        plt.plot(tspan, u_ref[i] * np.ones(len(tspan)), 'r--', linewidth=1.2)
        plt.grid()
        plt.xlim([tspan[0], tspan[-1]])
        plt.ylabel(r'{}'.format(u_label[i]))
        plt.title('Control trajectory')

    plt.show()
