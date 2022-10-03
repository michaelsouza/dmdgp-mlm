import numpy as np


def read_data():
    Y = []    
    return Y


def read_nmr(fn:str):
    D = {}
    with open(fn, 'r') as fid:
        for row in fid:
            s = row.split(',')
            i, j, d = int(s[0]), int(s[1]), float(s[2])
            if i not in D:
                D[i] = {}
            if j not in D:
                D[j] = {}
            D[i][j] = d
            D[j][i] = d
    return D


def read_csv(fn):
    x = []
    with open(fn, 'r') as fid:
        for row in fid:
            u = list(map(float, row.split(',')))
            x.append(u)
    return np.array(x)


def F(D: dict, x: np.ndarray):
    x = x.reshape(-1, 3)
    f = 0.0
    for i in sorted(D):
        for j in sorted(D[i]):
            if j > i:
                break
            fij = (x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2 + (x[i, 2] - x[j, 2])**2
            fij = fij - D[i][j]**2
            f += fij**2
    return f


def G(D: dict, x: np.ndarray):
    g = np.zeros(x.shape, dtype=float)
    # it easier to handle them when each row is an atom
    x = x.reshape(-1, 3)
    g = g.reshape(-1, 3)
    for i in sorted(D):
        for j in sorted(D[i]):
            if j > i: # avoid duplication
                break                        
            fij = (x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2 + (x[i, 2] - x[j, 2])**2 - D[i][j]**2
            for k in range(3):
                gik = 4 * (x[i, k] - x[j, k]) * fij
                g[i, k] += gik
                g[j, k] -= gik
    return g.reshape(-1)

