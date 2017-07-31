import sympy as sp
import sympy.physics.units as u
import numpy as np
import matplotlib.pyplot as plt
import logging as log


# Physical Quantity
class PQ:
    def __init__(self, val, dim=None, sigma=None, epsilon=None, symbol=None,
                 is_const=False):
        # if type(val) != u.quantities.Quantity:
        #    raise TypeError("Only sympy units can be used")
        if dim is not None:
            self.dim = dim

            self.val = u.convert_to(val, dim)

            if sigma is not None:
                self.sigma = u.convert_to(sigma, dim)
                self.epsilon = u.convert_to(self.sigma / self.val,
                                            sp.numbers.Integer(1))
            elif epsilon is not None:
                self.epsilon = u.convert_to(epsilon, sp.numbers.Integer(1))
                self.sigma = u.convert_to(self.val * self.epsilon, dim)
            else:
                if is_const:
                    self.sigma = self.epsilon = 0  # TODO:last digit
        else:
            # TODO: use dim of val
            self.val = val

            if sigma is not None:
                self.sigma = sigma
                self.epsilon = sigma / val
            elif epsilon is not None:
                self.epsilon = epsilon
                self.sigma = val * epsilon
            else:
                if is_const:
                    self.sigma = self.epsilon = 0

        if symbol is None:
            if not hasattr(PQ, 'symbol_counter'):
                PQ.symbol_counter = 0
            else:
                PQ.symbol_counter += 1
            self.symbol = sp.symbols('symbol' + str(PQ.symbol_counter))
        else:
            self.symbol = sp.symbols(symbol)

    # TODO: use internal dim information
    def convert(self, dim):
        return "%f±%f %s (%f%%)" % (
            float(u.convert_to(self.val, dim).n() / dim),
            float(u.convert_to(self.sigma, dim).n() / dim),
            dim,
            float(u.convert_to(self.epsilon, sp.numbers.Integer(1))))

    def raw_print(self):
        print(self.val)
        print(self.sigma)
        print(self.epsilon)

    def __str__(self):
        return self.convert(self.dim)

    def __repr__(self):
        return self.__str__()


def eval(dim, lambd, *args, symbol=None):
    partdiffs = [sp.diff(lambd(*[arg.symbol for arg in args]), x.symbol) for x
                 in args]
    log.debug('partdiffs %s', partdiffs)
    values = {(arg.symbol, arg.val) for arg in args}
    log.debug('values of partdiffs %s', values)
    summands = [sp.Abs(partdiff.subs(values).evalf() ** 2 * pe.sigma ** 2) for
                (partdiff, pe) in zip(partdiffs, args)]
    summands = [u.convert_to(s, dim ** 2) for s in summands]
    log.debug('summands %s', summands)
    new_val = u.convert_to(lambd(*[arg.val for arg in args]), dim)
    log.debug('val %s', new_val)
    new_sigma = u.convert_to(sp.sqrt(sum(summands)), dim)
    log.debug('sigma %s', new_sigma)
    return PQ(new_val, sigma=new_sigma, symbol=symbol, dim=dim)


def plt_pq(grid_exper, exper=None, grid_theor=None, theor=None):
    def get(lambd, arr):
        return np.array([lambd(elem) for elem in arr])

    if grid_theor is None:
        grid_theor = grid_exper
    for pqs, color, label, grid in ((exper, 'red', 'Экспериментальная', grid_exper),
                                    (theor, 'blue', 'Расчётная', grid_theor)):
        if pqs is not None:
            vals = get(lambda elem: elem.val, pqs)
            sigmas = get(lambda elem: elem.sigma, pqs)

            y = (vals / u.hz).astype(float)
            y_s = (sigmas / u.hz).astype(float)
            plt.plot(grid, y, color=color, label=label)
            plt.errorbar(grid, y, xerr=0, yerr=y_s, color=color)
            plt.scatter(grid, y, color=color)
    plt.grid()
    plt.legend()
