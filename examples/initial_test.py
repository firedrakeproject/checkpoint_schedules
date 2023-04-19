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

    def advance(self, n_0: int, n_1: int) -> None:
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        print((">"*(n_1-n_0)).rjust(n_1))
        i_n = n_0
        while i_n < n_1:
            i_np1 = i_n + 1
            i_n = i_np1
            # print(i_n)
        
        # if self.chk is None:
        #     counter = self.ic
        # else:
        #     counter = self.chk
        # self.ic = counter
        # while counter <= n_1:
        self.chk = i_n
            # counter += 1
           
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

    def advance(self, n_1: int, n_0: int, fwd_chk) -> None:
        """Execute the backward equation.

        Parameters
        ----------
        n1
            Initial time step in reverse state.
        n0
            Final time step in reverse state.

        """
        i_n = n_1
        while i_n > n_0:
            i_np1 = i_n - 1
            i_n = i_np1
        print("<".rjust(n_1))
            


fwd_ic = 0
steps = 10
schk = 3
fwd = Forward(steps, fwd_ic)
fwd.DefEquation()
bwd = Backward()
bwd.DefEquation()
manage = Manage(fwd, bwd, schk, steps)
manage.actions()
print("end")