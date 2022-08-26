# References:
# 1. 'Lavor, Carlile. "On generating instances for the molecular distance
#     geometry problem." Global optimization. Springer, Boston, MA, 2006. 405-414.
#

import os
import sys
import time
import numpy as np
from numpy.random import choice
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def createX(nnodes):
    rij = 1.526             # covalent bond length (in angstrons)
    tik = np.radians(109.5) # angle formed by two consecutive covalent bond
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


if __name__ == '__main__':
    np.random.seed(1)
    if len(sys.argv) < 3:
        print('Usage:\n> python createRandomInstance.py nsamples nnodes')
        raise Exception('Invalid arguments.')
    nsamples = int(sys.argv[1])
    nnodes = int(sys.argv[2])    
    print('Creating %d instances with %d nodes' % (nsamples, nnodes))
    # create DATA folder
    wdir = 'DATA_N%d_S%d' % (nnodes, nsamples)
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    # create instances
    tic = time.time()
    for i in tqdm(range(nsamples)):
        X = createX(nnodes)
        fcsv = os.path.join(wdir, 'pid_%04d.csv' % i)
        np.savetxt(fcsv, X=X, delimiter=',')
    toc = time.time() - tic
    print('Elapsed time %f secs' % toc)
