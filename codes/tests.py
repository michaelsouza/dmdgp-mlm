# References:
# 1. https://docs.python.org/3/library/unittest.html

from tkinter import SE
import unittest

from createRandomInstance import *


class TestCreateRandomInstance(unittest.TestCase):
    def test_createX(self):
        np.random.seed(1)
        nnodes = 10
        X = createX(nnodes)
        u = X[0] - X[1]
        v = X[2] - X[1]
        uv = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        cik_ref = uv/(norm_u * norm_v)
        rij_ref = np.linalg.norm(u)
        for i in range(2, len(X)):
            rij = np.linalg.norm(X[i] - X[i-1])
            self.assertAlmostEqual(rij_ref, rij)

            u = X[i-2] - X[i-1]
            v = X[i] - X[i-1]
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            cik = uv/(norm_u * norm_v)
            self.assertAlmostEqual(cik_ref, cik)


if __name__ == '__main__':
    unittest.main()
