import numpy as np

class Actuator:
    def __init__(self):
        self.w_n = 100 # default value
        self.damping_ratio = 0.5 # default value
        self.A = np.array([[0, 1],
                           [self.w_n ** 2, -2 * self.w_n * self.damping_ratio]]) # system matrix
        self.B = np.array([[0],
                           [self.w_n ** 2]])
    def dynamics(self, u_act, t, u_ctrl):
        '''
        :param u_act: control input after passing the actuator [y y_dot].T
        :param t: simulation time
        :param u_ctrl: control input before passing the actuator
        :return: derivative of y, i.e., [y_dot y_ddot].T
        '''
        return (np.dot(self.A, u_act) + np.dot(self.B, u_ctrl).squeeze()).squeeze()

