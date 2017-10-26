from .pq import *
import sympy as sp
import pandas as pd
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter

class PQLatexPrinter(LatexPrinter):
    #printmethod = '_print'

    def _print_PQ(self, expr):
        return 'oloprint'

    def _latex_PQ(self, expr):
        return 'ololatex'


class PQStrPrinter(StrPrinter):
    #printmethod = '_print'

    def _print_Quantity(self, expr):
        return "%s" % expr.abbrev


def prepare_to_print(df):
    df2 = pd.DataFrame()
    for column in df.columns:
        col = np.array(df[column].dropna().values)
        res_arr = np.array([np.array(x.str_rounded_as_decompose()) for x in col])
        # print(res_arr)

        # TODO: надо как-то проверять, что степень одна и та же, и контролировать это
        if len(res_arr[0]) == 4:
            # df[column].apply(lambda x: )
            df2['$%s$, %s'%(column, res_arr[0][2])] = pd.Series(res_arr[:, 0])
            # pd.Series(pqarray(df[column].dropna().values).val_float)
            df2['$\\sigma$($%s$), %s'%(column, res_arr[0][2])] = pd.Series(res_arr[:, 1])
            df2['$\\varepsilon$($%s$), %%'%(column)] = pd.Series(res_arr[:, 3])
        else:
            df2['$%s$, $10^{%s}$%s'%(column, res_arr[0][2], res_arr[0][3])] = pd.Series(res_arr[:, 0])
            df2['$\\sigma$($%s$), %s'%(column, res_arr[0][3])] = pd.Series(res_arr[:, 1])
            df2['$\\epsilon$($%s$), %%'%(column)] = pd.Series(res_arr[:, 4])
    return df2
