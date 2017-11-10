import sympy as sp
import sympy.physics.units as u
import numpy as np
import pandas as pd
import logging as log
from .arrays import *
from functools import total_ordering


# TODO: pretty printing
@total_ordering
class PQ:
    eps = 10e-10

    @staticmethod
    def __get_valid_args__(val):
        log.debug("val.args in __get_valid_args__: %s"%str(val.args))
        log.debug("with types: %s"%str([type(elem) for elem in val.args]))
        return [elem for elem in val.args if type(elem) != u.dimensions.Dimension]

    @staticmethod
    def get_dim(val):
        if not hasattr(val, 'args') or is_numeral_type(type(val)):
            return 1
        else:
            return np.prod([elem for elem in PQ.__get_valid_args__(val)
                            if not is_numeral_type(type(elem))])

    @staticmethod
    def get_value(val):
        if not hasattr(val, 'args') or is_numeral_type(type(val)):
            return val
        else:
            return np.prod([elem for elem in PQ.__get_valid_args__(val)
                            if is_numeral_type(type(elem))])

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

        if isinstance(val, PQ):
            raise Exception("Не пытайтесь передать PQ как val или sigma. Явно пропишите к нему .val")

        self.dim = None
        if dim is not None:
            self.dim = dim
        else:
            if val != 0:
                self.dim = PQ.get_dim(val)
            elif sigma is not None:
                self.dim = PQ.get_dim(sigma)
            else:
                raise Exception('Impossible to deduce dim (note: zero is dimensionless)')

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

        self.is_const = is_const

        if symbol is None:
            if not hasattr(PQ, 'symbol_counter'):
                PQ.symbol_counter = 0
            else:
                PQ.symbol_counter += 1
            self.symbol = sp.symbols('symbol' + str(PQ.symbol_counter))
        else:
            self.symbol = sp.symbols(symbol)

    # TODO
    # @property
    # def sigma(self):
    #     return self.val
    #
    # @sigma.setter
    # def sigma(self, sigma):
    #     self.__dict__['sigma'] = u.convert_to(sigma, self.dim)
    #     self.__dict__['epsilon'] = u.convert_to(self.sigma/self.val, sp.numbers.Integer(1))
    #
    # @property
    # def epsilon(self):
    #     return self.epsilon
    #
    # @epsilon.setter
    # def epsilon(self, epsilon):
    #     self.__dict__['epsilon'] = u.convert_to(epsilon, sp.numbers.Integer(1))
    #     self.__dict__['sigma'] = u.convert_to(self.val*self.epsilon, self.dim)

    def set_sigma(self, sigma):
        self.sigma = u.convert_to(sigma, self.dim)
        self.epsilon = u.convert_to(self.sigma/self.val, sp.numbers.Integer(1))
        return self

    def set_epsilon(self, epsilon):
        self.epsilon = u.convert_to(epsilon, sp.numbers.Integer(1))
        self.sigma = u.convert_to(self.val*self.epsilon, self.dim)
        return self

    def add_sigma(self, other_sigma):
        self.sigma = sp.sqrt(self.sigma**2 + other_sigma**2)
        self.epsilon = u.convert_to(self.sigma/self.val, sp.numbers.Integer(1))
        return self

    def repr_as(self, dim):
        """
        Конвертитует себя в размерность dim.
        :returns себя
        """
        self.dim = dim
        self.val = u.convert_to(self.val, dim).n()
        self.sigma = u.convert_to(self.sigma, dim).n()
        return self

    @staticmethod
    def get_float(val, dim):
        if val == sp.zoo:
            return np.nan

        #return float(u.convert_to(val, dim).n()/dim)
        return float(PQ.get_value(val))


    def str_as(self, dim=None):
        if dim is None:
            dim = self.dim

        return "%f±%f %s (%f)"%(
            PQ.get_float(self.val, dim),
            PQ.get_float(self.sigma, dim),
            dim,
            PQ.get_float(self.epsilon, 1))

    def raw_print(self):
        """Для дебага."""
        print(self.val)
        print(self.sigma)
        print(self.dim)
        print(self.epsilon)

    def __str__(self):
        return self.str_rounded_as(self.dim)

    def __repr__(self):
        return self.__str__()

    def str_rounded_as(self, dim=None, params=None):
        decomposition = self.str_rounded_as_decompose(dim, params=params)

        if len(decomposition) == 5:
            return '(%s±%s)*10^%s %s (%s%%)'%decomposition
        elif len(decomposition) == 4:
            return '%s±%s %s (%s%%)'%decomposition
        else:
            return '%f %s'%decomposition

    @staticmethod
    def __most_significant_digit(x):
        return int(sp.floor(sp.log(sp.Abs(x), 10))) + 1

    def get_print_params(self, dim=None):
        if dim is None:
            dim = self.dim
        float_sigma = PQ.get_float(self.sigma, dim)

        msd = self.__most_significant_digit(float_sigma)
        if float_sigma/10**(msd - 2) < 30:
            num_sign_dig = 2
        else:
            num_sign_dig = 1

        return msd, num_sign_dig

    def str_rounded_as_decompose(self, dim=None, params=None):
        """
        Правила округления в целом взяты из лабника.
        Процентов печатаются всегда первые две значащие цифры (с правильным округлением).
        :returns Строка, в которой величина записана в соответствии с правилами округления.
        """
        if dim is None:
            dim = self.dim

        if self.is_const == True:
            float_val = PQ.get_float(self.val, dim)
            return (float_val, '' if dim == 1 else dim)

        float_val = PQ.get_float(self.val, dim)
        float_sigma = PQ.get_float(self.sigma, dim)
        float_percents = PQ.get_float(self.epsilon, sp.numbers.Integer(1))*100

        log.debug("%f %f %f"%(float_val, float_sigma, float_percents))

        # def get_significant_digits(x, n):
        #     return round(x, n - self.__most_significant_digit(x) - 1)

        def round_to_precision(val, prec):
            return round(val/10**(prec))*10**(prec)

        # Порядок первой значащей цифры
        if params is None:
            msd, num_sign_dig = self.get_print_params(dim=dim)
        else:
            msd, num_sign_dig = params

        if not np.isnan(float_percents):
            msd_percents = self.__most_significant_digit(float_percents)
            str_percents = '%*.*f'%(max(msd_percents, 1), 2 - msd_percents, round_to_precision(float_percents, msd_percents - 2))
        else:
            str_percents = 'NaN'

        # print(type(msd))
        # print(num_sign_dig)
        # print(msd_percents)
        # Если значащая цифра настолько мала или велика, что "некрасиво", пишем в экспоненциальном виде.
        if msd > 3 or msd < -2:
            return (
                '%*.*f'%(1, num_sign_dig, round(float_val/10**(msd - num_sign_dig))/10),
                '%*.*f'%(1, num_sign_dig, round(float_sigma/10**(msd - num_sign_dig))/10),
                '%d'%(msd - num_sign_dig + 1),
                '%s'%('' if dim == 1 else dim),
                str_percents
            )
        else:
            return (
                '%*.*f'%(min(num_sign_dig, msd),
                         max(0, num_sign_dig - msd),
                         round_to_precision(float_val, msd - num_sign_dig)),
                '%*.*f'%(num_sign_dig if msd - num_sign_dig >= 0 else msd,
                         max(0, num_sign_dig - msd),
                         round_to_precision(float_sigma, msd - num_sign_dig)),
                '%s'%('' if dim == 1 else dim),
                str_percents
            )

    def __neg__(self):
        return eval(self.dim, lambda self: -self, self)

    def __add__(self, other):
        if issubclass(type(other), np.ndarray):
            return other + self
        return eval(self.dim, lambda self, other: self + other, self, other)

    def __radd__(self, other):
        return eval(self.dim, lambda self, other: self + other, self, other)

    def __sub__(self, other):
        if issubclass(type(other), np.ndarray):
            return -other + self
        return eval(self.dim, lambda self, other: self - other, self, other)

    def __rsub__(self, other):
        return eval(self.dim, lambda self, other: self - other, self, other)

    def __mul__(self, other):
        if issubclass(type(other), np.ndarray):
            return other*self
        if type(other) is PQ:
            new_dim = self.dim*other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim*PQ.get_dim(other)
        else:
            new_dim = self.dim

        return eval(new_dim, lambda self, other: self*other, self, other)

    def __rmul__(self, other):
        if type(other) is PQ:
            new_dim = self.dim*other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim*PQ.get_dim(other)
        else:
            new_dim = self.dim

        return eval(new_dim, lambda self, other: self*other, self, other)

    def __truediv__(self, other):
        if issubclass(type(other), np.ndarray):
            return 1/other*self
        if type(other) is PQ:
            new_dim = self.dim/other.dim
        elif hasattr(other, 'args'):
            new_dim = self.dim/PQ.get_dim(other)
        else:
            new_dim = self.dim

        return eval(new_dim, lambda self, other: self/other, self, other)

    def __rtruediv__(self, other):
        if type(other) is PQ:
            new_dim = self.dim/other.dim
        elif hasattr(other, 'args'):
            new_dim = PQ.get_dim(other)/self.dim
        else:
            new_dim = 1/self.dim

        return eval(new_dim, lambda self, other: other/self, self, other)

    def __pow__(self, power, modulo=None):
        if not is_numeral_type(type(power)):
            raise Exception('Тип степени %s. Возводить в степень, которая не число, нельзя.'%type(power))

        return eval(self.dim**power, lambda self, other: self**power, self, power)

    def sqrt(self):
        return self**(sp.numbers.Rational(1, 2))

    def log(self):
        # TODO: какой-то баг мешает конвертировать физические величины с логарифмами.
        # Workaround: возвращать безразмерную
        # return PQ(sp.log(PQ.get_val_from_args(self.val))*sp.log(PQ.get_dim_from_args(self.val)),
        #           sigma=sp.log(PQ.get_val_from_args(self.sigma))*sp.log(PQ.get_dim_from_args(self.sigma)))
        return PQ(sp.log(PQ.get_value(self.val)),
                  sigma=PQ.get_value(self.sigma)/PQ.get_value(self.val))

    def __lt__(self, other):
        if type(other) is not PQ:
            raise Exception("Can compare PQ only with PQ")
        if self.dim != other.dim:
            raise Exception("Can compare PQ only of same dim")
        if self.val < other.val:
            return True
        else:
            return False

    def __eq__(self, other):
        if type(other) is not PQ:
            raise Exception("Can compare PQ only with PQ")
        if self.dim != other.dim:
            raise Exception("Can compare PQ only of same dim")
        if self.val == other.val:
            return True
        else:
            return False

    def _print(self, expr=None):
        return 'clprint'

    def _latex(self, expr=None):
        return 'cllatex'  # '$'+self.__str__()+'$'#


def is_numeral_type(t):
    return t in {int, float, np.float64, sp} or issubclass(t, sp.numbers.Number)


# TODO: Узнавать, какая величина давала наибольший вклад в погрешность.
def eval(dim, lambd, *args, symbol=None):
    """
    Вычисляет новую PE по формуле и пересчитывает погрешности как надо.
    :param dim: Размерность желаемой величины.
    :param lambd: Функция, которая будет вычисляться. Обычно в виде лямбды.
    :param args: Параметры, которые надо передавать функции.
    :param symbol: можно особо указать sympy символ (зачем?)
    :return: Новое PQ.
    """
    args = list(args)
    log.debug('args: %s', args)

    log.debug("Initial args types: %s"%str([type(arg) for arg in args]))
    for i in range(len(args)):
        if issubclass(type(args[i]), np.ndarray):
            if type(args[i][0]) is not PQ:
                args[i] = pqarray([PQ(val, is_const=True) for val in args[i]])
        elif type(args[i]) is not PQ:
            args[i] = PQ(args[i], is_const=True)
    log.debug('args converted to PQ: %s', args)

    partdiffs = [sp.diff(lambd(*[arg.symbol for arg in args]), x.symbol) for x in args]
    log.debug('partdiffs %s', partdiffs)

    values = {(arg.symbol, arg.val) for arg in args}
    log.debug('values of partdiffs %s', values)

    summands = [sp.Abs(partdiff.subs(values).evalf()**2*pe.sigma**2) for
                (partdiff, pe) in zip(partdiffs, args)]
    summands = [u.convert_to(s, dim**2) for s in summands]
    log.debug('summands %s', summands)

    new_val = u.convert_to(lambd(*[arg.val for arg in args]), dim).n()

    log.debug('val %s', new_val)

    new_sigma = u.convert_to(sp.sqrt(sum(summands)), dim).n()
    log.debug('sigma %s', new_sigma)

    log.debug('dim %s', dim)

    return PQ(new_val, sigma=new_sigma, symbol=symbol, dim=dim)


def celsium_to_kelvins(c):
    return (c + 273.15)*u.kelvins


def get_mean(arr):
    if type(arr) is list or type(arr) is pd.Series:
        arr = pqarray(arr)
    mean = np.mean(arr.val)
    n = len(arr)
    separate_error = (1/(n - 1)*np.sum((arr.val - mean)**2))**sp.numbers.Rational(1, 2)
    mean_error = separate_error/np.sqrt(n)
    return PQ(np.asscalar(mean), sigma=sp.sqrt(np.asscalar(mean_error)**2 + arr[0].sigma**2))
