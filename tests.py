import unittest
from miptlabs import *


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


if __name__ == '__main__':
    unittest.main()
