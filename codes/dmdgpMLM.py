import os
import random
import numpy as np
from typing import Callable
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

class MLM:
    def __init__(self) -> None:
        self.k = 0
        self.B = None
        self.Rx = None
        self.Ry = None
        self.distX = None
        self.distY = None
        self.c = None
        self.g = None
        self.dyEst = None

    def training(self, X: np.ndarray, Y: np.ndarray, k: int, distX: Callable[[np.ndarray, np.ndarray], float], distY: Callable[[np.ndarray, np.ndarray], float], idx: list = [], seed: int = 1) -> None:
        if len(X) != len(Y):
            raise Exception('X and Y must have the same length.')

        # TODO RMSD dos y's de referência

        n = X.shape[0]
        if len(idx) == 0:
            # set random seed for reproducibility
            random.seed(seed) 
            idx = random.sample(range(n))
        self.Rx = X[idx] # reference points in X
        self.Ry = Y[idx] # reference points in Y
        Dx = np.zeros((k, n), dtype=float)
        Dy = Dx.copy()
        for i in range(k):
            for j in range(n):
                Dx[i, j] = distX(self.Rx[i], X[j])
                Dy[i, j] = distY(self.Ry[i], Y[j])
        self.k = len(self.Rx)
        self.setB(Dx, Dy)
        self.distY = distY
        self.distX = distX

    def setB(self, Dx: np.ndarray, Dy: np.ndarray) -> None:
        ''' Calc B by solving Dx.T @ Dx @ B = Dx.T @ Dy, 
        using QR decomposition'''
        
        # If Q, R = qr(Dx), then
        #    Dx.T @ Dx @ B = Dx.T @ Dy
        #    R.T @ Q.T @ Q @ R @ B = R.T @ Q.T @ Dy
        #    R.T @ R @ B = R.T @ Q.T @ Dy
        #    B = R^{-1} @ (R.T)^-1 @ R.T @ Q.T @ Dy
        #    B = R^{-1} @ Q.T @ Dy

        Q, R = np.linalg.qr(Dx) # Q @ R = Dx
        self.B = Q.T @ Dy
        solve_triangular(R, self.B, trans='N', lower=False, overwrite_b=True)

    def test(self, x:np.ndarray) -> np.ndarray:
        dx = np.zeros((self.k,), dtype=float)
        for i in range(self.k):
            dx[i] = self.distX(x, self.Rx[i])
        self.dyEst = (dx @ self.B)**2
        y = np.mean(self.Ry) # initial value
        self.c = y.copy()
        self.g = y.copy()
        J = lambda y: self.Jfun(y)
        r = minimize(J, y, jac=bool)
        if not r.success:
            raise Exception('The minimize method has failed.')
        y = r.x
        return y

    def Jfun(self, y: np.ndarray):        
        f = 0.0
        for i in range(self.k):
            dyi2 = self.distY(y, self.Ry[i])**2            
            self.c[i] = dyi2 - self.dyEst[i]**2
            f += self.c[i]**2
        for j in range(y.shape[0]):
            self.g[j] = 0.0
            for i in range(self.k):
                self.g[j] += self.c[i] * (y[j] - self.Ry[i][j])
            self.g[j] *= 4.0
        return f, self.g
        

if __name__ == "__main__":
    wdir = 'DATA_N50_S10'
    random.seed(1)
    ds = sorted([fn for fn in os.listdir(wdir) if fn.endswith('.nmr')])
    ds_train = sorted(random.sample(ds, int(0.8 * len(ds))))
    ds_test = sorted(list(set(ds) - set(ds_train)))
    print(ds_train)
    print(ds_test)
    
