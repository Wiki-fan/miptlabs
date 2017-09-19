import sympy as sp
import sympy.physics.units as u
import numpy as np
import matplotlib.pyplot as plt
import logging as log


# TODO: pretty printing
class PQ:
    eps = 10e-5

    def get_dim_from_args(val):
        return np.prod([elem for elem in val.args
                        if type(elem) != u.dimensions.Dimension and not PQ.is_numeral_type(type(elem))])

    def get_from_array(lambd, arr):
        return np.array([lambd(elem) for elem in arr])

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

        if type(val) is PQ:
            raise Exception("Не пытайтесь передать PQ как val или sigma. Явно пропишите к нему .val")

        if dim is not None:
            self.dim = dim
        elif not hasattr(val, 'args'):
            self.dim = 1
        else:
            self.dim = PQ.get_dim_from_args(val)

        self.val = u.convert_to(val, self.dim)

        if sigma is not None:
            self.sigma = u.convert_to(sigma, self.dim)
            self.epsilon = u.convert_to(self.sigma/self.val, sp.numbers.Integer(1))
        elif epsilon is not None:
            self.epsilon = u.convert_to(epsilon, sp.numbers.Integer(1))
            self.sigma = u.convert_to(self.val*self.epsilon, self.dim)
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
        new_dim = None
        if type(other) is PQ:
            new_dim = self.dim*other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim*PQ.get_dim_from_args(other)
        else:
            new_dim = self.dim

        return eval(new_dim,
                    lambda self, other: self*other, self, other)

    def __rmul__(self, other):
        new_dim = None
        if type(other) is PQ:
            new_dim = self.dim*other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim*PQ.get_dim_from_args(other)
        else:
            new_dim = self.dim

        return eval(new_dim,
                    lambda self, other: self*other, self, other)

    def __truediv__(self, other):
        new_dim = None
        if type(other) is PQ:
            new_dim = self.dim/other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim/PQ.get_dim_from_args(other)
            print('derived_dim', new_dim, other.args)
        else:
            new_dim = self.dim

        return eval(new_dim,
                    lambda self, other: self/other, self, other)

    def __rtruediv__(self, other):
        new_dim = None
        if type(other) is PQ:
            new_dim = self.dim/other.dim
        elif hasattr(other, 'args'):
            new_dim = PQ.get_dim_from_args(other)/self.dim
        else:
            new_dim = 1/self.dim

        return eval(new_dim,
                    lambda self, other: other/self, self, other)

    def __pow__(self, power, modulo=None):
        if not PQ.is_numeral_type(type(power)):
            raise Exception('Тип степени %s. Возводить в степень, которая не число, нельзя.'%type(power))

        return eval(self.dim**power,
                    lambda self, other: self**power, self, power)
    
    def is_numeral_type(t):
        return t in {int, float, np.float64, sp.numbers.Integer, sp}

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
        if type(args[i]) is not PQ:
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


def get_nparray_from_PQs(pqs):
    vals = PQ.get_from_array(lambda elem: elem.val, pqs)
    sigmas = PQ.get_from_array(lambda elem: elem.sigma, pqs)
    x = (vals/pqs[0].dim).astype(float)
    x_s = (sigmas/pqs[0].dim).astype(float)
    return (x, x_s)


def plt_pq(grid, values, label=None, color=None, ols=False, grid_x=None,
           grid_y=None, plot=plt.plot):
    """
    Строит графики. С крестами погрешностей. ols=True рисует ещё и прямую, приближающую значения по МНК.
    Вызовы plt.figure и plt.show должны быть снаружи.
    Можно добавлять подписи к осям и прочее.
    """

    if type(grid[0]) is PQ:
        x, x_s = get_nparray_from_PQs(grid)
    else:
        x = grid
        x_s = 0

    if type(values[0]) is PQ:
        y, y_s = get_nparray_from_PQs(values)
    else:
        y = values
        y_s = 0

    line = plot(x, y, color=color, label=label, zorder=2)
    plt.errorbar(x, y, xerr=x_s, yerr=y_s, color=line[0].get_color(), zorder=3)
    plt.scatter(x, y, color=line[0].get_color(), zorder=4, alpha=0.2)

    ax = plt.axes()
    if grid_y is not None:
        try:
            length = len(grid_y)
        except:
            length = 1

        if length > 1:
            major = grid_y[0]
            minor = grid_y[1]
        else:
            major = grid_y
            minor = grid_y/5

        yticks_major = np.arange(plt.ylim()[0], plt.ylim()[1] + PQ.eps, major)
        ax.set_yticks(yticks_major)
        if minor != 0:
            yticks_minor = np.arange(plt.ylim()[0], plt.ylim()[1] + PQ.eps, minor)
            ax.set_yticks(yticks_minor, minor=True)

    if grid_x is not None:
        try:
            length = len(grid_x)
        except:
            length = 1
        if length > 1:
            major = grid_x[0]
            minor = grid_x[1]
        else:
            major = grid_x
            minor = grid_x/5
        xticks_major = np.arange(plt.xlim()[0], plt.xlim()[1] + PQ.eps, major)
        ax.set_xticks(xticks_major)
        if minor != 0:
            xticks_minor = np.arange(plt.xlim()[0], plt.xlim()[1] + PQ.eps, minor)
            ax.set_xticks(xticks_minor, minor=True)
    ax.grid(axis='both', color='black')
    ax.grid(axis='both', which='minor', color='gray')

    if label is not None:
        plt.legend()

    if type(grid[0]) is PQ:
        plt.xlabel(ax.get_xlabel() + str(grid[0].dim))
    if type(values[0]) is PQ:
        plt.ylabel(ax.get_ylabel() + str(values[0].dim))

    if ols == True:
        ols_coefs, ols_errors = OLS(grid, values)
        x1 = x[0] - PQ.eps
        x2 = x[-1] + PQ.eps
        y1 = ols_coefs[0]*x1 + ols_coefs[1]
        y2 = ols_coefs[0]*x2 + ols_coefs[1]
        plot([x1, x2], [y1, y2], color='black',
                 linestyle='dashed', zorder=5,
                 label='OLS for %s'%label)


def OLS(grid, values):
    """
    Коэффициенты и погрешности начиная с наивысшей степени.
    """
    if type(grid[0]) is PQ:
        x, x_s = get_nparray_from_PQs(grid)
    else:
        x = grid
        x_s = 0

    if type(values[0]) is PQ:
        y, y_s = get_nparray_from_PQs(values)
    else:
        y = values
        y_s = 0
    coefs = np.polyfit(x, y, deg=1)
    sigma_b = 1/np.sqrt(len(y))*np.sqrt((np.mean(y**2) - np.mean(y)**2)/(np.mean(x**2) - np.mean(x)**2) - coefs[0]**2)
    sigma_a = sigma_b*(np.mean(x**2) - np.mean(x)**2)
    errors = [sigma_b, sigma_a]
    return coefs, errors


def celsium_to_kelvins(c):
    return (c + 273.15)*u.kelvins


def repr_ndarray_as(arr, dim):
    arr = np.array([val.repr_as(dim) for val in arr])
    return arr
