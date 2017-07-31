import unittest
from miptlabs import *


class TestPE(unittest.TestCase):
    def test_init_and_str(self):
        distance = PQ(1 * u.m, sigma=1 * u.mm)
        distance2 = PQ(1 * u.m, epsilon=1e-5)
        print(distance)
        print(distance2)


if __name__ == '__main__':
    unittest.main()
