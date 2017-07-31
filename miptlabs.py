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
        # if type(val) != u.quantities.Quantity:
        #    raise TypeError("Only sympy units can be used")
        if dim is not None:
            self.dim = dim
        else:
            self.dim = np.prod(val.args[1:])

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

        if symbol is None:
            if not hasattr(PQ, 'symbol_counter'):
                PQ.symbol_counter = 0
            else:
                PQ.symbol_counter += 1
            self.symbol = sp.symbols('symbol' + str(PQ.symbol_counter))
        else:
            self.symbol = sp.symbols(symbol)

    # TODO: use internal dim information
    def repr_as(self, dim=None):
        if dim is None:
            dim = self.dim

        return "%f±%f %s (%f)" % (
            float(u.convert_to(self.val, dim).n() / dim),
            float(u.convert_to(self.sigma, dim).n() / dim),
            dim,
            float(u.convert_to(self.epsilon, sp.numbers.Integer(1))))

    def raw_print(self):
        print(self.val)
        print(self.sigma)
        print(self.epsilon)

    def __str__(self):
        return self.repr_as(self.dim)

    def __repr__(self):
        return self.__str__()

    # TODO
    def repr_rounded_as(self, dim=None):
        if dim is None:
            dim = self.dim

        float_val = float(u.convert_to(self.val, dim).n() / dim)
        float_sigma = float(u.convert_to(self.sigma, dim).n() / dim)
        float_percents = float(u.convert_to(self.epsilon, sp.numbers.Integer(1)))*100

        # print(float_val, float_sigma, float_percents)

        def most_significant_digit(x):
            return int(sp.floor(sp.log(sp.Abs(x), 10))) + 1

        def get_significant_digits(x, n):
            return round(x, n - most_significant_digit(x) - 1)

        # Если первые значащие цифры погрешности 1 или 2, оставляем 2 цифры, иначе 1
        msd = most_significant_digit(float_sigma)
        num_sign_dig = None
        if float_sigma / 10**(msd - 2) < 30:
            num_sign_dig = 2
        else:
            num_sign_dig = 1
        # print(msd)
        # print(num_sign_dig)
        # msd_percents = most_significant_digit(float_epsilon) TODO: решить, что делать здесь
        # print(msd_percents)
        return '%*.*f±%*.*f %s (%f%%)' % (
            num_sign_dig if msd - num_sign_dig >= 0 else msd,
            0 if msd - num_sign_dig >= 0 else num_sign_dig - msd,
            round(float_val / 10**(msd - num_sign_dig)) * 10**(msd - num_sign_dig),
            num_sign_dig if msd - num_sign_dig >= 0 else msd,
            0 if msd - num_sign_dig >= 0 else num_sign_dig - msd,
            round(float_sigma / 10**(msd - num_sign_dig)) * 10**(msd - num_sign_dig),
            dim,
            float_percents)

# TODO: автоустановление dim по dim'ам величин
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
    partdiffs = [sp.diff(lambd(*[arg.symbol for arg in args]), x.symbol) for x
                 in args]
    log.debug('partdiffs %s', partdiffs)
    values = {(arg.symbol, arg.val) for arg in args}
    log.debug('values of partdiffs %s', values)
    summands = [sp.Abs(partdiff.subs(values).evalf()**2 * pe.sigma**2) for
                (partdiff, pe) in zip(partdiffs, args)]
    summands = [u.convert_to(s, dim**2) for s in summands]
    log.debug('summands %s', summands)
    new_val = u.convert_to(lambd(*[arg.val for arg in args]), dim)
    log.debug('val %s', new_val)
    new_sigma = u.convert_to(sp.sqrt(sum(summands)), dim)
    log.debug('sigma %s', new_sigma)
    return PQ(new_val, sigma=new_sigma, symbol=symbol, dim=dim)


def plt_pq(grid_exper, exper=None, grid_theor=None, theor=None):
    """
    Строит графики. Рассчитана на график экспериментальных и теоретических значений, с крестами погрешностей.
    По умолчанию используется grid_exper, если теоретических данных другое количество, надо задавать grid_theor.
    Вызовы plt.figure и plt.show должны быть снаружи.
    Можно добавлять подписи к осям и прочее.
    """
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
