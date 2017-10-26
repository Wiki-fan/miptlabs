import matplotlib.pyplot as plt
from .pq import *
from .arrays import *
import pandas as pd

def plt_pq(grid, values, label=None, color=None, ols=False, grid_x=None,
           grid_y=None, plot=plt.plot):
    """
    Строит графики. С крестами погрешностей. ols=True рисует ещё и прямую, приближающую значения по МНК.
    Вызовы plt.figure и plt.show должны быть снаружи.
    Можно добавлять подписи к осям и прочее.
    """

    def get_arr_and_sigmas(values):
        if type(values) is pd.Series:
            values = pqarray(values)
        if type(values[0]) is PQ:
            return values.val_float, values.sigma_float, values
        else:
            return values, 0, values

    x, x_s, grid = get_arr_and_sigmas(grid)
    y, y_s, values = get_arr_and_sigmas(values)

    line = plot(x, y, color=color, label=label, zorder=2)
    plt.errorbar(x, y, xerr=x_s, yerr=y_s, color=line[0].get_color(), zorder=3)
    plt.scatter(x, y, color=line[0].get_color(), zorder=4, alpha=0.2)

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


def plot_OLS(grid, values, plot=plt.plot, label=None):
    """Построить график прямой, приближающей точки по НМК, на текущей figure.
    :returns (ols_coefs, ols_errors)"""
    ols_coefs, ols_errors = OLS(grid, values)
    if type(grid[0]) is PQ:
        x, x_s = grid.val_float, grid.sigma_float
    else:
        x = grid
        x_s = 0
    x1 = np.min(x)
    x2 = np.max(x)
    y1 = ols_coefs[0]*x1 + ols_coefs[1]
    y2 = ols_coefs[0]*x2 + ols_coefs[1]

    plot([x1, x2], [y1, y2], color='black',
         linestyle='dashed', zorder=5,
         label='OLS for %s'%label)

def OLS(grid, values):
    """
    :returns Коэффициенты и погрешности начиная с наивысшей степени.
    """
    grid = pqarray(grid)
    values = pqarray(values)
    if type(grid[0]) is PQ:
        x, x_s = grid.val_float, grid.sigma_float
    else:
        x = grid
        x_s = 0

    if type(values[0]) is PQ:
        y, y_s = values.val_float, values.sigma_float
    else:
        y = values
        y_s = 0
    coefs = np.polyfit(x, y, deg=1)
    sigma_b = 1/np.sqrt(len(y))*np.sqrt((np.mean(y**2) - np.mean(y)**2)/(np.mean(x**2) - np.mean(x)**2) - coefs[0]**2)
    sigma_a = sigma_b*(np.mean(x**2) - np.mean(x)**2)
    errors = [sigma_b, sigma_a]
    return coefs, errors

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
