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
        plt.xlabel('Time')
        plt.ylabel(r'{}'.format(u_label[i]))
        plt.title('Control trajectory')

def plot_P(Matrix_list):
    iters = len(Matrix_list)
    m = Matrix_list[0].shape[0]

    Matrix_hist = np.zeros((iters, m, m))
    for iter in range(iters):
        for i in range(m):
            for j in range(m):
                Matrix_hist[iter, i, j] = Matrix_list[iter][i, j]

    plt.figure()
    for i in range(m):
        for j in range(i, m):
            plt.plot(range(iters), Matrix_hist[:, i, j], linewidth=1.2, label=r'P({}, {})'.format(i+1, j+1))
    plt.xlim([0, iters-1])
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend(loc='upper right')

def plot_K(Matrix_list):
    iters = len(Matrix_list)
    m = Matrix_list[0].shape[0]
    n = Matrix_list[0].shape[1]

    Matrix_hist = np.zeros((iters, m, n))
    for iter in range(iters):
        for i in range(m):
            for j in range(n):
                Matrix_hist[iter, i, j] = Matrix_list[iter][i, j]

    plt.figure()
    for i in range(m):
        for j in range(i, n):
            plt.plot(range(iters), Matrix_hist[:, i, j], linewidth=1.2, label=r'K({}, {})'.format(i+1, j+1))
    plt.xlim([0, iters-1])
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend(loc='upper right')

def plot_w(w_hist, tspan):
    num_w = w_hist.shape[0]

    plt.figure()
    for i in range(num_w):
        plt.plot(tspan, w_hist[i, :], linewidth=1.2, label=r'w({})'.format(i+1))
        plt.grid()
        plt.xlim([tspan[0], tspan[-1]])
        plt.title('Weight History')
    plt.legend(loc='upper right')

def plot_cond(cond_list):
    t = range(len(cond_list))
    plt.figure()
    plt.plot(t, cond_list)
    plt.yscale('log', base=10)
    plt.grid()