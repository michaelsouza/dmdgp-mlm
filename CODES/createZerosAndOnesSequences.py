import os
import sys
import numpy as np
from tqdm import tqdm


def createZerosAndOnesSequences(fcsv):
    x = np.loadtxt(fcsv, delimiter=',')
    nnodes = x.shape[0]
    s = np.zeros(nnodes, dtype=bool)
    for i in range(3, nnodes):
        a = x[i-3]
        b = x[i-2]
        c = x[i-1]
        d = x[i]
        u = a - c
        v = b - c
        w = np.cross(u, v)
        p = d - c
        s[i] = np.dot(w, p) > 0
    fseq = fcsv.replace('.csv', '.seq')
    np.savetxt(fseq, s, delimiter=',', fmt='%d')


if __name__ == '__main__':
    wdir = sys.argv[1]
    print("Creating 0's and 1's sequences on folder %s" % wdir)
    for fcsv in tqdm(sorted(os.listdir(wdir))):
        if not fcsv.endswith('.csv'):
            continue
        fcsv = os.path.join(wdir, fcsv)
        createZerosAndOnesSequences(fcsv)
