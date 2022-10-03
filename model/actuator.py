import numpy as np

class Actuator:
    def __init__(self):
        self.w_n = 1 # default value
        self.damping_ratio = 0.5 # default value
        self.A = np.array([[0, 1],
                           [self.w_n ** 2, -2 * self.w_n * self.damping_ratio]]) # system matrix
        self.B = np.array([[0],
                           [self.w_n ** 2]])
    def dynamics(self, y, t, y_c):
        '''
        :param y: control input after passing the actuator [y y_dot].T
        :param t: simulation time
        :param y_c: control input before passing the actuator
        :return: derivative of y, i.e., [y_dot y_ddot].T
        '''
        return (np.dot(self.A, y) + np.dot(self.B, y_c)).squeeze()

