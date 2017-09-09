import unittest
from miptlabs import *
import miptlabs as ml


class TestPE(unittest.TestCase):
    def test_print_rounded(self):
        a = PQ(12345.6*u.m, sigma=3*u.m, dim=u.m)
        print(a.repr_rounded_as(u.m))
        b = PQ(12345.4*u.m, sigma=3*u.m, dim=u.m)
        print(b.repr_rounded_as(u.m))
        c = PQ(0.4*u.m, sigma=3*u.m, dim=u.m)
        print(c.repr_rounded_as(u.m))
        d = PQ(0.4*u.m, sigma=0.03*u.m, dim=u.m)
        print(d.repr_rounded_as(u.m))

    def test_opers(self):
        a = ml.PQ(1*u.m, dim=u.m, sigma=1*u.mm)
        b = ml.PQ(2*u.m, dim=u.m, sigma=1*u.mm)
        c = ml.PQ(2*u.m, dim=u.m, sigma=1*u.mm)
        # type(u.m)
        # u.convert_to(a.sigma, a.dim)
        ml.eval(a.dim, lambda a, b: a + b, a, b)
        # a+b #3.000000±0.001414 meter (0.000471)
        # (a*3) # 3.000000±0.003000 meter (0.001000)
        # a*b # 2.000000±0.002236 meter**2 (0.001118)
        # a/b # 0.500000±0.000559 1 (0.001118)
        # 1/b # 0.500000±0.000250 1/meter (0.000500)
        # type(u.m.args[1])

if __name__ == '__main__':
    unittest.main()
