import sympy as sp
__all__ = \
    [
        "Backward"
    ]

class Backward():
    """This object define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.sol = None

    def DefEquation(self):
        """Define the symbolic equation.

        """
        # Create a symbol x
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.exp = (x - y)
    def Advance(self, n_1: int, n_0: int, fwd_chk) -> None:
        """Execute the backward equation.

        Parameters
        ----------
        n1
            Initial time step in reverse state.
        n0
            Final time step in reverse state.

        """
        for s in range(n_1, n_0, -1):
            # out = self.exp.subs("x", s)
            out = self.exp.subs([("x", s), ("y", fwd_chk)])
            print(out)
            
            
 

