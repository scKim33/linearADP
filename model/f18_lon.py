import numpy as np
from scipy.io import loadmat


class f18_lon:
    def __init__(
            self,
            x0=None,
            x_ref=None,
            C=np.array(([1, 0, 0, 0], [0, 0, 0, 1])),
            Q=np.diag([1, 100, 10, 100]),
            R=np.diag([1e6, 1e6]),
            Qa=np.diag([1, 100, 10, 100, 1, 0, 0, 1]),
            Ra=np.diag([1e6, 1e6])
    ):
        # initial x_ref setting from trim condition
        self.x_trim = loadmat('../dat/f18_lin_data.mat')['x_trim_lon'].squeeze()
        # x_ref default value : x_trim
        noise = np.array([np.random.normal(0, 0.1 * self.x_trim[0]),
                          np.random.normal(0, 0.1 * self.x_trim[1]),
                          np.random.normal(0, 1e-3),
                          np.random.normal(0, 1e-3)])
        # x0 default value : x_trim + noise
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = noise
            
        if x_ref is not None:
            self.x_ref = x_ref
        else:
            self.x_ref = np.zeros(self.x0.shape)
        self.mat = loadmat('../dat/f18_lin_data.mat')
        self.A = self.mat['Alon']
        self.B = self.mat['Blon']
        self.C = C
        self.Q = Q
        self.R = R
        self.Aa = np.block([[self.A, np.zeros((np.shape(self.A)[0], np.shape(self.C)[0]))],
                            [self.C, np.zeros((np.shape(self.C)[0], np.shape(self.C)[0]))]])
        self.Ba = np.block([[self.B],
                            [np.zeros((np.shape(self.C)[0], np.shape(self.B)[1]))]])
        self.Qa = Qa
        self.Ra = Ra

    def dynamics(self, x, t, u):
        return (np.dot(self.A, x) + np.dot(self.B, u)).squeeze()

    def aug_dynamics(self, x, t, u):
        aug_x = np.block(
            [[np.reshape(x, (len(x), 1))]])  # x is already augmented at test_scalar.py, so is not required to add zeros
        aug_x_ref = np.block([[np.zeros((np.shape(self.A)[0], 1))],
                              [np.reshape(self.x_ref,
                                          (len(self.x_ref), -1))]])  # x_ref shape : (# row of C,) -> (# row of C, 1)
        return np.dot(self.Aa, aug_x).squeeze() + np.dot(self.Ba, u).squeeze() - aug_x_ref.squeeze()
