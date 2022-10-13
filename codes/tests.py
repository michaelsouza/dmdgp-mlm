# References:
# 1. https://docs.python.org/3/library/unittest.html

import os
import unittest
import numpy as np
from dmdgp_ML import *
from scipy.optimize import check_grad, approx_fprime

def check_grad2(f, g, x):    
    fold = f(x)
    gold = g(x)
    xnew = x.copy()
    gnew = gold.copy()
    h = 1E-8
    for i in range(len(x)):
        xnew[i] = x[i] + h        
        gnew[i] = (f(xnew) - fold) / h
        xnew[i] = x[i]
    err = np.max(np.abs(gnew - gold))
    return err

class TestAll(unittest.TestCase):
    def test_F(self):
        wdir = '/home/michael/gitrepos/dmdgp-mlm/DATA_D6_N50_S10/'
        for fn in os.listdir(wdir):            
            if not fn.endswith('.nmr'):
                continue
            fn = os.path.join(wdir, fn)            
            numNodes, numEdges, D = read_nmr(fn)
            x = read_csv(fn.replace('.nmr','.csv'))
            y = F(x, D)
            self.assertAlmostEqual(y, 0)
    
    def test_G(self):
        np.random.seed(1)        
        wdir = '/home/michael/gitrepos/dmdgp-mlm/DATA_D6_N50_S10/'
        x = read_csv(os.path.join(wdir, 'pid_0000.csv'))
        x = x.reshape(-1) # flattening
        FILES = [fn for fn in os.listdir(wdir) if fn.endswith('.nmr')]
        for k, fn in enumerate(FILES):
            fn = os.path.join(wdir, fn)
            numNodes, numEdges, D = read_nmr(fn)                        
            f = lambda x: F(x, D)
            g = lambda x: G(x, D)
            gx_num = approx_fprime(x, f, 1E-8)
            gx = g(x)                        
            r = np.linalg.norm(gx_num - gx) / np.max((1, np.linalg.norm(gx_num)))            
            self.assertAlmostEqual(r, 0, 3)



if __name__ == '__main__':
    unittest.main()
