import sympy as sp
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
        return "%s" % expr.abbrev #expr.abbrev
