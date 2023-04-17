from manage import Manage
import sympy as sp

class Forward():
    """Define the a forward solver.

    """
    def __init__(self, steps, initial_condition):
        self.exp = None
        self.chk_id = None
        self.steps = steps
        self.ic = initial_condition
        self.chk = None

    def DefEquation(self):
        """Define the symbolic equation.

        """
        # Create a symbol x
        x = sp.symbols("x")
        self.exp = x

    def Advance(self, n_0: int, n_1: int) -> None:
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        if self.chk is None:
            counter = self.ic
        else:
            counter = self.chk
        self.ic = counter
        while counter <= n_1:
            self.chk = self.exp.subs("x", counter)
            counter += 1
           
    def GetSteps(self) -> int:
        """Return the total time steps.

        """
        return self.steps
   
    def UpdateInitCondition(self, data) -> None:
        """Clear the initial condition data.

        """
        self.chk = data


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
            assert out == 0


fwd_ic = 0
steps = 10
schk = 3
fwd = Forward(steps, fwd_ic)
fwd.DefEquation()
bwd = Backward()
bwd.DefEquation()
manage = Manage(fwd, bwd, schk, steps)
manage.actions()
