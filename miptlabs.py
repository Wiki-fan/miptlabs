import sympy as sp
import sympy.physics.units as u
import numpy as np
import matplotlib.pyplot as plt
import logging as log


# TODO: pretty printing
class PQ:
    def __init__(self, val, dim=None, sigma=None, epsilon=None, symbol=None,
                 is_const=False):
        """
        Physical Quantity
        :param val: значение
        :param dim: размерность величины. Если указана, используется она. Если None, берётся размерность val.
        :param sigma: относительная погрешность
        :param epsilon:  абсолютная погрешность
            Стоит указывать только одну погрешность, другая посчитается сама.
        :param symbol: можно особо указать sympy символ (зачем?)
        :param is_const: позволяет сделать константу, у которой не будет погрешностей.
            Иначе же за погрешность надо брать последнюю цифру (не реализовано).
        """

        if dim is not None:
            self.dim = dim
        elif PQ.is_simple_type(val):
            self.dim = 1
        else:
            self.dim = np.prod([elem for elem in val.args[1:] if type(elem) != u.dimensions.Dimension])

        self.val = u.convert_to(val, self.dim)

        if sigma is not None:
            self.sigma = u.convert_to(sigma, dim)
            self.epsilon = u.convert_to(self.sigma/self.val, sp.numbers.Integer(1))
        elif epsilon is not None:
            self.epsilon = u.convert_to(epsilon, sp.numbers.Integer(1))
            self.sigma = u.convert_to(self.val*self.epsilon, dim)
        else:
            if is_const:
                self.sigma = self.epsilon = 0
            else:
                pass  # TODO:last digit

        if symbol is None:
            if not hasattr(PQ, 'symbol_counter'):
                PQ.symbol_counter = 0
            else:
                PQ.symbol_counter += 1
            self.symbol = sp.symbols('symbol' + str(PQ.symbol_counter))
        else:
            self.symbol = sp.symbols(symbol)

    def repr_as(self, dim):
        self.dim = dim
        self.val = u.convert_to(self.val, dim).n()
        self.sigma = u.convert_to(self.sigma, dim).n()
        return self

    def str_as(self, dim=None):
        if dim is None:
            dim = self.dim

        return "%f±%f %s (%f)"%(
            float(u.convert_to(self.val, dim).n()/dim),
            float(u.convert_to(self.sigma, dim).n()/dim),
            dim,
            float(u.convert_to(self.epsilon, 1)))

    def raw_print(self):
        print(self.val)
        print(self.sigma)
        print(self.epsilon)

    def __str__(self):
        return self.str_as(self.dim)

    def __repr__(self):
        return self.__str__()

    # TODO
    def repr_rounded_as(self, dim=None):
        if dim is None:
            dim = self.dim

        float_val = float(u.convert_to(self.val, dim).n()/dim)
        float_sigma = float(u.convert_to(self.sigma, dim).n()/dim)
        float_percents = float(u.convert_to(self.epsilon, sp.numbers.Integer(1)))*100

        # print(float_val, float_sigma, float_percents)

        def most_significant_digit(x):
            return int(sp.floor(sp.log(sp.Abs(x), 10))) + 1

        def get_significant_digits(x, n):
            return round(x, n - most_significant_digit(x) - 1)

        # Если первые значащие цифры погрешности 1 или 2, оставляем 2 цифры, иначе 1
        msd = most_significant_digit(float_sigma)
        num_sign_dig = None
        if float_sigma/10**(msd - 2) < 30:
            num_sign_dig = 2
        else:
            num_sign_dig = 1
        # print(msd)
        # print(num_sign_dig)
        # msd_percents = most_significant_digit(float_epsilon) TODO: решить, что делать здесь
        # print(msd_percents)
        return '%*.*f±%*.*f %s (%f%%)'%(
            num_sign_dig if msd - num_sign_dig >= 0 else msd,
            0 if msd - num_sign_dig >= 0 else num_sign_dig - msd,
            round(float_val/10**(msd - num_sign_dig))*10**(msd - num_sign_dig),
            num_sign_dig if msd - num_sign_dig >= 0 else msd,
            0 if msd - num_sign_dig >= 0 else num_sign_dig - msd,
            round(float_sigma/10**(msd - num_sign_dig))*10**(msd - num_sign_dig),
            dim,
            float_percents)

    def __add__(self, other):
        return eval(self.dim, lambda self, other: self + other, self, other)

    def __radd__(self, other):
        return eval(self.dim, lambda self, other: self + other, self, other)

    def __sub__(self, other):
        return eval(self.dim, lambda self, other: self - other, self, other)

    def __rsub__(self, other):
        return eval(self.dim, lambda self, other: self - other, self, other)

    def __mul__(self, other):
        return eval(self.dim if PQ.is_simple_type(other) else self.dim*other.dim,
                    lambda self, other: self*other, self, other)

    def __rmul__(self, other):
        return eval(self.dim if PQ.is_simple_type(other) else self.dim*other.dim,
                    lambda self, other: self*other, self, other)

    def __truediv__(self, other):
        return eval(self.dim if PQ.is_simple_type(other) else self.dim/other.dim,
                    lambda self, other: self/other, self, other)

    def __rtruediv__(self, other):
        return eval(1/self.dim if PQ.is_simple_type(other) else other.dim/self.dim,
                    lambda self, other: other/self, self, other)

    def __pow__(self, power, modulo=None):
        return eval(self.dim**power if PQ.is_simple_type(power) else self.dim**power.dim,
                    lambda self, other: self**power, self, power)

    def is_simple_type(arg):
        return type(arg) is int or type(arg) is float or type(arg) is np.float64 \
               or type(arg) is u.quantities.Quantity #or type(arg) is sympy.Mul
        #return type(arg) is not PQ and type(arg) is not np.ndarray


# TODO: Узнавать, какая величина давала наибольший вклад в погрешность.
def eval(dim, lambd, *args, symbol=None):
    """
    Вычисляет новую PE по формуле и пересчитывает погрешности как надо.
    :param dim: Желаемая величина
    :param lambd: Функция, которая будет вычисляться. Обычно в виде лямбды.
    :param args: Параметры, которые надо передавать функции. Обязательно должны быть PQ.
        Если нужны какие-нибудь другие параметры, например int, следует оформлять их как частичное применение функции
        или как-нибудь ещё.
    :param symbol: можно особо указать sympy символ (зачем?)
    :return: Новое PQ.
    """
    args = list(args)
    log.debug('args: %s', args)

    for i in range(len(args)):
        if PQ.is_simple_type(args[i]):
            args[i] = PQ(args[i], is_const=True)
    log.debug('args converted to PQ: %s', args)

    partdiffs = [sp.diff(lambd(*[arg.symbol for arg in args]), x.symbol) for x
                 in args]
    log.debug('partdiffs %s', partdiffs)

    values = {(arg.symbol, arg.val) for arg in args}
    log.debug('values of partdiffs %s', values)

    summands = [sp.Abs(partdiff.subs(values).evalf()**2*pe.sigma**2) for
                (partdiff, pe) in zip(partdiffs, args)]
    summands = [u.convert_to(s, dim**2) for s in summands]
    log.debug('summands %s', summands)

    new_val = u.convert_to(lambd(*[arg.val for arg in args]), dim)
    log.debug('val %s', new_val)

    new_sigma = u.convert_to(sp.sqrt(sum(summands)), dim)
    log.debug('sigma %s', new_sigma)

    return PQ(new_val, sigma=new_sigma, symbol=symbol, dim=dim)


def plt_pq(grid, values, label=None, color=None, ols=False):
    """
    Строит графики. С крестами погрешностей. ols=True рисует ещё и прямую, приближающую значения по НМК.
    Вызовы plt.figure и plt.show должны быть снаружи.
    Можно добавлять подписи к осям и прочее.
    """

    def get(lambd, arr):
        return np.array([lambd(elem) for elem in arr])

    vals = get(lambda elem: elem.val, values)
    sigmas = get(lambda elem: elem.sigma, values)

    y = (vals/values[0].dim).astype(float)
    y_s = (sigmas/values[0].dim).astype(float)
    line = plt.plot(grid, y, color=color, label=label, zorder=1)
    plt.errorbar(grid, y, xerr=0, yerr=y_s, color=line[0].get_color(), zorder=1)
    plt.scatter(grid, y, color=line[0].get_color(), zorder=2)

    if ols == True:
        ols_coefs, ols_errors = OLS(grid, values)
        x1 = grid[0]-1
        x2 = grid[-1] + 1
        y1 = ols_coefs[0]*x1 + ols_coefs[1]
        y2 = ols_coefs[0]*x2 + ols_coefs[1]
        plt.plot([x1, x2], [y1, y2], color='black',
                 linestyle='dashed', zorder=3,
                 label='OLS for %s'%label if label is not None else None)

    plt.grid()
    plt.legend()


def OLS(x, y):
    if type(y[0]) is PQ:
        y = np.array([float(pq.val/y[0].dim) for pq in y])
    coefs = np.polyfit(x, y, deg=1)
    sigma_b = 1/np.sqrt(len(y))*np.sqrt((np.mean(y**2)-np.mean(y)**2)/(np.mean(x**2)-np.mean(x)**2)-coefs[0]**2)
    sigma_a = sigma_b*(np.mean(x**2)-np.mean(x)**2)
    errors = [sigma_b, sigma_a]
    return coefs, errors

def celsium_to_kelvins(c):
    return (c+273.15)*u.kelvins
