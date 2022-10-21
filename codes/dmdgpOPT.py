import time
import numpy as np
from scipy.optimize import minimize


def read_nmr(fn: str):
    D = []
    numEdges, numNodes = 0, 0
    with open(fn, 'r') as fid:
        for row in fid:
            s = row.split(',')
            i, j, dij = int(s[0]), int(s[1]), float(s[2])
            D.append((i, j, dij))
            numEdges += 1
            numNodes = np.max([i, j, numNodes])
    numNodes += 1
    return numNodes, numEdges, sorted(D)


def read_csv(fn):
    x = []
    with open(fn, 'r') as fid:
        for row in fid:
            u = list(map(float, row.split(',')))
            x.append(u)
    return np.array(x)


def F(x: np.ndarray, D: list):
    x = x.reshape(-1, 3)
    f = 0.0
    for i, j, dij in D:
        fij = (x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2 + \
            (x[i, 2] - x[j, 2])**2 - dij**2
        f += fij**2
    return f


def G(x: np.ndarray, D: list):
    g = np.zeros(x.shape, dtype=float)
    # it easier to handle them when each row is an atom
    x = x.reshape(-1, 3)
    g = g.reshape(-1, 3)
    for i, j, dij in sorted(D):
        fij = (x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2 + \
            (x[i, 2] - x[j, 2])**2 - dij**2
        for k in range(3):
            gik = 4 * (x[i, k] - x[j, k]) * fij
            g[i, k] += gik
            g[j, k] -= gik
    return g.reshape(-1)

def FG(x: np.ndarray, D:list):    
    x = x.reshape(-1, 3)    
    f = 0.0
    g = np.zeros(x.shape, dtype=float)
    g = g.reshape(-1, 3)
    for i, j, dij in D:
        fij = (x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2 + \
            (x[i, 2] - x[j, 2])**2 - dij**2
        f += fij**2
        for k in range(3):
            gik = 4 * (x[i, k] - x[j, k]) * fij
            g[i, k] += gik
            g[j, k] -= gik
    return f, g.reshape(-1)


def initX(numNodes: int, D: dict):
    x = np.zeros((numNodes, 3), dtype=float)
    for i, j, dij in D:
        u = np.random.uniform(low=-1, high=1, size=3)
        u = u / np.linalg.norm(u)
        x[j] = x[i] + dij * u
    return x


def calcDeviation(x: np.ndarray, D: dict):
    dev = []
    for i, j, dij in D:
        aij = np.linalg.norm(x[i] - x[j])
        dev.append(dij - aij)
    return dev


if __name__ == "__main__":
    fn = 'DATA_D6_N50_S10/pid_0000.nmr'
    numSamples = 10

    print('fn: %s' % fn)
    numNodes, numEdges, D = read_nmr(fn)
    print('numNodes: %d, numEdges: %d' % (numNodes, numEdges))
    np.random.seed(42)
    options = {'disp': True}

    for sample in range(numSamples):
        tic = time.time()
        print('sample: %d' % sample)
        x = initX(numNodes, D)
        print('F(x0): %g' % F(x, D))
        # res = minimize(F, x, args=(D), options=options)
        # res = minimize(F, x, jac=G, args=(D), options=options)
        res = minimize(FG, x, jac=True, args=(D), options=options)
        x = res.x.reshape((numNodes, -1))
        d = calcDeviation(x, D)
        print('max_abs(d): %g' % np.max(np.abs(d)))
        toc = time.time() - tic
        print('tElapsed (secs): %f' % toc)
