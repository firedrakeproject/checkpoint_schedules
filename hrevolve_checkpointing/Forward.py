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
           
    def GetTimesteps(self) -> int:
        """Return time steps.

        """
        return self.steps
   
    
    def ReadCheckpoint(self, data) -> None:
        """Update the initial Condition.
        
        Parameters
        ----------
        data
            Initial condition data.
        
        Notes
        -----
        This method is called for time > t_0, where t_0 is the initial time.
        
        """
        self.ic = data
    
    def ClearCheckpoint(self) -> None:
        """Clear the initial condition data.

        """
        self.chk = None

    def UpdateInitCondition(self, data) -> None:
        """Clear the initial condition data.

        """
        self.chk = data