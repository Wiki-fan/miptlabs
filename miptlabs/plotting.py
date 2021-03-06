try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('Matplotlib not found')

from .pq import *
from .arrays import *
import pandas as pd


@convert_args
def _get_arr_and_sigmas(values):
    if type(values[0]) is PQ:
        return values.val_float, values.sigma_float, values
    else:
        return values, 0, values


def plt_pq(grid, values, label=None, color=None, ols=False, grid_x=None,
           grid_y=None, plot=None, **kwargs):
    """
    Строит графики. С крестами погрешностей. ols=True рисует ещё и прямую, приближающую значения по МНК.
    Вызовы plt.figure и plt.show должны быть снаружи.
    Можно добавлять подписи к осям и прочее.
    """

    if plot is None:
        plot = plt.plot

    x, x_s, grid = _get_arr_and_sigmas(grid)
    y, y_s, values = _get_arr_and_sigmas(values)

    # Почему у plot alpha не работает?
    line = plot(x, y, alpha=0.1, color=color, label=label, zorder=2, **kwargs)
    plt.errorbar(x, y, xerr=x_s, yerr=y_s, color=line[0].get_color(), zorder=3, alpha=1)
    plt.scatter(x, y, color=line[0].get_color(), zorder=4, alpha=0.2, **kwargs)

    fig = plt.gcf()
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
    ax.grid(which='major', color='black', alpha=0.4)
    ax.grid(which='minor', color='black', alpha=0.7)

    if label is not None:
        plt.legend()

    # if type(grid[0]) is PQ:
    #     plt.xlabel(ax.get_xlabel() + str(grid[0].dim))
    # if type(values[0]) is PQ:
    #     plt.ylabel(ax.get_ylabel() + str(values[0].dim))

    if ols == True:
        plot_OLS(grid, values, plot, label)


def plot_OLS(x_, y_, plot=None, label=None):
    """Построить график прямой, приближающей точки по МНК, на текущей figure.
    :returns (ols_coefs, ols_errors)"""

    if plot is None:
        plot = plt.plot

    x = pqarray(x_)
    y = pqarray(y_)
    if type(x[0]) is PQ:
        x, x_s = x.val_float, x.sigma_float
    else:
        x = x
        x_s = 0
    ols_coefs, ols_errors = OLS(x, y)
    x1 = np.min(x)
    x2 = np.max(x)
    y1 = ols_coefs[0]*x1 + ols_coefs[1]
    y2 = ols_coefs[0]*x2 + ols_coefs[1]

    plot([x1, x2], [y1, y2], color='black',
         linestyle='dashed', zorder=5,
         label='OLS for %s'%label)

    return ols_coefs, ols_errors


def OLS(x_, y_, add_sigma=True):
    """
    :returns Коэффициенты и погрешности начиная с наивысшей степени.
    """
    x_ = pqarray(x_)
    y_ = pqarray(y_)
    if type(x_[0]) is PQ:
        x, x_s = x_.val_float, x_.sigma_float
    else:
        x = x_
        x_s = 0

    if type(y_[0]) is PQ:
        y, y_s = y_.val_float, y_.sigma_float
    else:
        y = y_
        y_s = 0
    coefs = np.polyfit(x, y, deg=1)
    sigma_b = 1/np.sqrt(len(y))*np.sqrt((np.mean(y**2) - np.mean(y)**2)/(np.mean(x**2) - np.mean(x)**2) - coefs[0]**2)
    sigma_a = sigma_b*(np.mean(x**2) - np.mean(x)**2)
    if add_sigma:
        sigma_b = np.sqrt(sigma_b**2+np.mean(x_s)**2)
        sigma_a = np.sqrt(sigma_a**2+np.mean(y_s)**2)
    errors = [sigma_b, sigma_a]

    return coefs, errors


def get_intersections_with_axes(a, b):
    """y = ax+b, x = y/a-b/a"""
    return -b/a, b


def get_intersection_of_lines(a1, b1, a2, b2):
    x = (b2 - b1)/(a1 - a2)
    return x, a1*x + b1

def get_intersection_of_lines_full(a1, b1, c1, a2, b2, c2):
    if a1 == 0:
        tmp = a1, b1, c1
        a1, b1, c1 = a2, b2, c2
        a2, b2, c2 = tmp
    y = (c2*a1-a2*c1)/(b1*a2-a1*b2)

    return (-c1-b1*y)/a1, y

# def graphical_errors(grid, values):
#     if type(grid[0]) is PQ:
#         x, x_s = grid.val_float, grid.sigma_float
#     else:
#         x = grid
#         x_s = 0
#
#     if type(values[0]) is PQ:
#         y, y_s = values.val_float, values.sigma_float
#     else:
#         y = values
#         y_s = 0
#
#     left = x[:len(x)/3]
#     left_s = x_s[:len(x)/3]
#
#     right = x[len(x)*2/3:]
#     right_s = x_s[len(x)*2/3:]
#
#     mid_x = x[len(x)/2]
#     mid_y = y[len(x)/2]
#
#     cons = ({'type':'ineq', })
#     f = lambda left, right: x
