# References:
# 1. 'Lavor, Carlile. "On generating instances for the molecular distance
#     geometry problem." Global optimization. Springer, Boston, MA, 2006. 405-414.
#

import os
import sys
import time
import itertools
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def createX(nnodes: int) -> np.ndarray:
    rij = 1.526             # covalent bond length (in angstrons)
    tik = np.radians(109.5)  # angle formed by two consecutive covalent bond
    # prefered angles
    W = [60, 180, 300]
    W = np.random.choice(W, size=nnodes, replace=True)
    # possible pertubations (is discrete, but it could be continuos)
    E = np.arange(-15, 16, 1)
    E = np.random.choice(E, size=nnodes, replace=True)
    # final perturbed angles(in radians)
    W = np.radians(W + E)

    # set initial base
    cik = np.cos(tik)
    sik = np.sin(tik)
    X = np.zeros((nnodes, 3), dtype=float)
    X[0] = (0, 0, 0)
    X[1] = (-rij, 0, 0)
    X[2] = (rij * cik - rij, rij * sik, 0)

    for i in range(3, nnodes):
        a = X[i-3]
        b = X[i-2]
        c = X[i-1]

        # create rotations from vectors (the vectors' norm are the rotation angles)
        v = b - c
        Rv = R.from_rotvec(v * (W[i] / np.linalg.norm(v)))
        u = np.cross(v, a - c)
        Ru = R.from_rotvec(u * (tik / np.linalg.norm(u)))

        # translate to origin and apply the rotations Ru and Rv
        x = Ru.apply(b - c)
        x = Rv.apply(x)

        # get final x
        X[i] = x + c
    return X


def createNMR(X: np.ndarray, dmax: float) -> np.ndarray:
    A = []
    for i, j in itertools.combinations(range(len(X)), 2):
        xi, xj = X[i], X[j]
        dij = np.linalg.norm(xi - xj)
        if dij <= dmax:
            A.append((i,j,dij))
    return np.array(A)

def createBinarySequence(X):
    nnodes = X.shape[0]
    s = np.zeros(nnodes, dtype=bool)
    for i in range(3, nnodes):
        a = X[i-3]
        b = X[i-2]
        c = X[i-1]
        d = X[i]
        u = a - c
        v = b - c
        w = np.cross(u, v)
        p = d - c
        s[i] = np.dot(w, p) > 0
    return s


if __name__ == '__main__':
    np.random.seed(1)
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-nsamples':
            nsamples = int(int(sys.argv[i+1]))
        elif arg == '-nnodes':
            nnodes = int(int(sys.argv[i+1]))
        elif arg == '-dmax':
            dmax = int(int(sys.argv[i+1]))
        else:
            raise Exception('Unkown argument (arg=%s)' % (str(arg)))
        i += 2
    print('Creating instances')
    print('   nsamples ... %d' % nsamples)
    print('   nnodes ..... %d' % nnodes)
    print('   dmax ....... %g' % dmax)
    # create DATA folder
    wdir = 'DATA_D%g_N%d_S%d' % (dmax, nnodes, nsamples)
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    # create instances
    tic = time.time()
    for i in tqdm(range(nsamples)):
        X = createX(nnodes)
        A = createNMR(X, dmax)
        s = createBinarySequence(X)
        fcsv = os.path.join(wdir, 'pid_%04d.csv' % i)
        np.savetxt(fcsv, X=X, fmt='%.16g', delimiter=',')
        fnmr = fcsv.replace('.csv','.nmr')
        np.savetxt(fnmr, X=A, fmt='%.16g', delimiter=',')
        fseq = fcsv.replace('.csv', '.seq')
        np.savetxt(fseq, s, delimiter=',', fmt='%d')

    toc = time.time() - tic
    print('Elapsed time %f secs' % toc)
