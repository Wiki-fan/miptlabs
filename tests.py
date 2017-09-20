import unittest
from miptlabs import *
import miptlabs as ml

a = ml.pq(1*u.m, dim=u.meter, sigma=1*u.mm) # Почему meter/m не считается?
b = ml.pq(2*u.m, sigma=1*u.mm)
c = ml.pq(2*u.m, sigma=1*u.mm)
t = ml.pq(2*u.s, sigma=0.01*u.s)

print((b/u.s))

class TestPE(unittest.TestCase):
    def test_print_rounded(self):
        a = PQ(12345.6*u.m, sigma=3*u.m, dim=u.m)
        print(a.str_rounded_as(u.m))
        b = PQ(12345.4*u.m, sigma=3*u.m, dim=u.m)
        print(b.str_rounded_as(u.m))
        c = PQ(0.4*u.m, sigma=3*u.m, dim=u.m)
        print(c.str_rounded_as(u.m))
        d = PQ(0.4*u.m, sigma=0.03*u.m, dim=u.m)
        print(d.str_rounded_as(u.m))

    def test_opers(self):
        a = ml.pq(1*u.m, dim=u.m, sigma=1*u.mm)
        b = ml.pq(2*u.m, dim=u.m, sigma=1*u.mm)
        c = ml.pq(2*u.m, dim=u.m, sigma=1*u.mm)
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
