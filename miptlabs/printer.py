from .pq import *
import sympy as sp
import pandas as pd
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter


class PQLatexPrinter(LatexPrinter):
    # printmethod = '_print'

    def _print_PQ(self, expr):
        return 'oloprint'

    def _latex_PQ(self, expr):
        return 'ololatex'


class PQStrPrinter(StrPrinter):
    # printmethod = '_print'

    def _print_Quantity(self, expr):
        return "%s"%expr.abbrev


# TODO: тестирование!
def prepare_to_print(df):
    df2 = pd.DataFrame()
    for column in df.columns:
        col = pqarray(df[column].dropna().values)

        if col[0].is_const == True:
            res_arr = np.array([np.array(x.str_rounded_as_decompose()) for x in col])
        else:
            params = col.common_print_params()
            res_arr = np.array([np.array(x.str_rounded_as_decompose(params=params)) for x in col])
        # print(res_arr)

        if len(res_arr[0]) == 4:
            # df[column].apply(lambda x: )
            df2['\\thead{$%s$, \\\\ %s}'%(column, res_arr[0][2])] = pd.Series(res_arr[:, 0])
            # pd.Series(pqarray(df[column].dropna().values).val_float)
            df2['\\thead{$\\sigma$($%s$), \\\\ %s}'%(column, res_arr[0][2])] = pd.Series(res_arr[:, 1])
            df2['\\thead{$\\varepsilon$($%s$), \\\\ %%}'%(column)] = pd.Series(res_arr[:, 3])
        elif len(res_arr[0]) == 5:
            df2['\\thead{$%s$, \\\\ $10^{%s}$%s}'%(column, res_arr[0][2], res_arr[0][3])] = pd.Series(res_arr[:, 0])
            df2['\\thead{$\\sigma$($%s$), \\\\ %s}'%(column, res_arr[0][3])] = pd.Series(res_arr[:, 1])
            df2['\\thead{$\\epsilon$($%s$), \\\\ %%}'%(column)] = pd.Series(res_arr[:, 4])
        else:
            # TODO: красивое выравнивание
            df2['\\thead{$%s$, \\\\ %s}'%(column, res_arr[0][1])] = pd.Series(res_arr[:, 0])
    return df2


def write_latex(file, table_to_print):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(r"""\documentclass[russian]{article}
\usepackage{amsgen, amsmath, amstext, amsbsy, amsopn, amsfonts, amsthm, thmtools,  amssymb, amscd, mathtext, mathtools}
\usepackage[T1, T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{makecell}
\usepackage[a4paper,left=10mm,right=10mm,top=10mm,bottom=10mm]{geometry}
\setlength{\tabcolsep}{.16667em}
\begin{document}
\begin{landscape}""")
        n = len(table_to_print.columns)
        f.write(table_to_print.to_latex(encoding='utf-8', escape=False, na_rep='', )
                .replace('%', '\%').replace('l'*(n + 1), '|' + 'l|'*(n + 1)))
        f.write(r"""\end{landscape}\end{document}""")
