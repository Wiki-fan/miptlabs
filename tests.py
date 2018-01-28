import unittest

import sympy.physics.units as u
import sympy as sp

import numpy as np
import logging as log
import functools
import matplotlib.pyplot as plt
import seaborn

import importlib.util
#spec = importlib.util.spec_from_file_location("miptlabs", "/full/path/to/miptlabs.py")
#miptlabs = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(miptlabs)
#ml = miptlabs

import miptlabs as ml
importlib.reload(ml)

import sys, importlib
importlib.reload(log)
log.basicConfig(level=log.INFO, format='%(message)s', stream=sys.stdout)

#sp.latex=lambda expr, **settings:ml.PQLatexPrinter(settings).doprint(expr)
sp.Basic.__str__=lambda expr, **settings:ml.PQStrPrinter(settings).doprint(expr)
#sp.Basic.__str__=lambda expr, **settings:ml.PQLatexPrinter(settings).doprint(expr)
#sp.init_printing(latex_printer=ml.PQLatexPrinter)
#sp.init_printing()
#sp.Basic.__str__ = lambda self: ml.PQLatexPrinter().doprint(self)

class TestStringMethods(unittest.TestCase):

    def test_dimension_of_symbols(self):
        t1 = ml.PQ(1.0*u.s, is_const=True)
        self.assertEquals(t1.dim, u.s)
        t2 = ml.PQ(1*u.s, is_const=True)
        self.assertEquals(t2.dim, u.s)
        # possible workaround: use floats everywhere!

    def test_decimal_multipliers(self):
        x = ml.PQ(2.0*u.milli*u.m, sigma=0.1*u.milli*u.m)

    def test_format(self):
        a = ml.PQ(12345.6*u.m, sigma=3*u.m, dim=u.m)
        self.assertEqual(str(a), '12346±3 m (0.024%)')
        a1 = ml.PQ(12345.5*u.m, sigma=3*u.m, dim=u.m)
        self.assertEqual(str(a1), '12346±3 m (0.024%)')
        b = ml.PQ(12345.4*u.m, sigma=3*u.m, dim=u.m)
        self.assertEqual(str(b), '12345±3 m (0.024%)')
        c = ml.PQ(0.4*u.m, sigma=0.3*u.m, dim=u.m)
        self.assertEqual(str(c), '0.4±0.3 m (75%)')
        d = ml.PQ(0.4*u.m, sigma=0.03*u.m, dim=u.m)
        self.assertEqual(str(d), '0.40±0.03 m (7.5%)')
        e = ml.PQ(0.00004*u.m, sigma=0.000003*u.m)
        self.assertEqual(str(e), '(4.0±0.3)*10^-5 m (7.5%)')
        f = ml.PQ(123456*u.m, sigma=3000*u.m)
        self.assertEqual(str(f), '(12.3±0.3)*10^4 m (2.4%)')
        g = ml.PQ(100000*u.m, sigma=3*u.m)
        self.assertEqual(str(g), '100000±3 m (0.0030%)')
        h = ml.PQ(1.0*u.m, sigma=1.0*u.mm)
        self.assertEqual(str(h), '1.0000±0.0010 m (0.10%)')

    def test_arithmetic(self):
        a = ml.PQ(1.0*u.m, sigma=1.0*u.mm)
        b = ml.PQ(2.0*u.m, sigma=1.0*u.mm)
        c = ml.PQ(3.0*u.m, sigma=1.0*u.mm)

        self.assertEqual(str(a+b), '3.0000±0.0014 m (0.047%)')
        self.assertEqual(str(a*3), '3.000±0.003 m (0.10%)')
        self.assertEqual(str(a*b), '2.0000±0.0022 m**2 (0.11%)')
        self.assertEqual(str(a/b), '(500.0±0.6)*10^-3  (0.11%)')
        # Была двойка в размерности, починено, оставлено для предотвращения регрессии
        self.assertEqual(str((u.s/b).val), '0.5*s/m')
        self.assertEqual(str(1/b), '(5000.00±2.50)*10^-4 1/m (0.050%)')

        self.assertEqual(str(-a).replace('\n', ''),
                         '-1.0000±0.0010 m (0.10%)')
        self.assertEqual(str(b + c), '5.0000±0.0014 m (0.028%)')
        self.assertEqual(str(c + b), '5.0000±0.0014 m (0.028%)')
        self.assertEqual(str(c - b), '1.0000±0.0014 m (0.14%)')
        self.assertEqual(str(b - c), '-1.0000±0.0014 m (0.14%)')
        self.assertEqual(str(c*b).replace('\n', ''), '6.000±0.004 m**2 (0.060%)')
        self.assertEqual(str(b*c).replace('\n', ''), '6.000±0.004 m**2 (0.060%)')
        self.assertEqual(str(b/c).replace('\n', ''), '(666.7±0.4)*10^-3  (0.060%)')
        self.assertEqual(str(c/b), '(1500.0±0.9)*10^-3  (0.060%)')
        self.assertEqual(str(b**2), '4.000±0.004 m**2 (0.10%)')
        self.assertEqual(str(np.sqrt(b)).replace('\n', ''), '(1414.2±0.4)*10^-3 sqrt(m) (0.025%)')

    def test_array(self):
        a = ml.PQ(1.0*u.m, sigma=1.0*u.mm)
        b = ml.PQ(2.0*u.m, sigma=1.0*u.mm)
        c = ml.PQ(2.0*u.m, sigma=1.0*u.mm)

        arr = ml.pqarray([a, b, c])

        self.assertEqual(str(arr),
                         '[1.0000±0.0010 m (0.10%) 2.0000±0.0010 m (0.050%) 2.0000±0.0010 m (0.050%)]')
        # Проверка, что там с размерностью у нуля
        self.assertEqual(str((arr - a)[0].dim), 'm')
        self.assertEqual(str(-arr).replace('\n', ''),
                         '[-1.0000±0.0010 m (0.10%) -2.0000±0.0010 m (0.050%) -2.0000±0.0010 m (0.050%)]')
        self.assertEqual(str(a + arr), '[2.0000±0.0028 m (0.14%) 3.0000±0.0014 m (0.047%) 3.0000±0.0014 m (0.047%)]')
        self.assertEqual(str(arr + a), '[2.0000±0.0028 m (0.14%) 3.0000±0.0014 m (0.047%) 3.0000±0.0014 m (0.047%)]')
        self.assertEqual(str(a - arr), '[0.0000±0.0014 m (NaN%) -1.0000±0.0014 m (0.14%) -1.0000±0.0014 m (0.14%)]')
        self.assertEqual(str(arr - a), '[0.0000±0.0014 m (NaN%) 1.0000±0.0014 m (0.14%) 1.0000±0.0014 m (0.14%)]')
        self.assertEqual(str(b*arr).replace('\n', ''),
                         '[2.0000±0.0022 m**2 (0.11%) 4.000±0.006 m**2 (0.14%) 4.0000±0.0028 m**2 (0.071%)]')
        self.assertEqual(str(arr*b).replace('\n', ''),
                         '[2.0000±0.0022 m**2 (0.11%) 4.000±0.006 m**2 (0.14%) 4.0000±0.0028 m**2 (0.071%)]')
        self.assertEqual(str(b/arr).replace('\n', ''),
                         '[2.0000±0.0022  (0.11%) (1000.0±0.7)*10^-3  (0.071%) (1000.0±0.7)*10^-3  (0.071%)]')
        self.assertEqual(str(arr/b), '[(500.0±0.6)*10^-3  (0.11%) 1.00±0.00  (0.00%) (1000.0±0.7)*10^-3  (0.071%)]')
        self.assertEqual(str(b**[1, 2, 3]),
                         '[2.0000±0.0010 m (0.050%) 4.000±0.004 m**2 (0.10%) 8.000±0.012 m**3 (0.15%)]')
        self.assertEqual(str(arr**2).replace('\n', ''),
                         '[1.0000±0.0028 m**2 (0.28%) 4.000±0.006 m**2 (0.14%) 4.000±0.006 m**2 (0.14%)]')
        self.assertEqual(str(np.sqrt(arr)).replace('\n', ''),
                         '[(1000.0±0.5)*10^-3 sqrt(m) (0.050%) (1414.2±0.4)*10^-3 sqrt(m) (0.025%) (1414.2±0.4)*10^-3 sqrt(m) (0.025%)]')

    def test_types(self):
        with self.assertRaises(Exception) as e:
            a = ml.PQ(1.0*u.m, sigma=1.0*u.mm)
            test = ml.PQ(a, sigma=2*u.m)

        #self.assertEqual(e.exception.msg,
        #                 'Не пытайтесь передать PQ как val или sigma. Явно пропишите к нему .val')

if __name__ == '__main__':
    unittest.main()
