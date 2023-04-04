import sympy as sp

class Backward():
    """This object define the a forward solver.

    """
    def __init__(self):
        self.exp = None

    def def_equation(self):
        """Define the symbolic equation.

        """
        # Create a symbol x
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.exp = x + 1 - y
      

    def advance(self, n_1: int, n_0: int) -> None:
        """Advance the backward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        for s in range(n_0, n_1):
            out = self.exp.subs("x", s)
            # out = self.exp.subs([("x", s), ("y", chk)])
            

    # def read_checkpoint(self) -> None:
    #     """Verify if is checkpointed.
        
    #     Parameters
    #     ----------
    #     is_checkpointed
    #         If `True`, the checkpoint data is saved.

    #     """
    #     print("bwd")
        # if is_checkpointed:
        #     self.func.get_checkpoint_id(n_write)
        #     assert n_write==self.n_0
        #     self.func.save_checkpoint(self.output_0)
        #     print(self.func.chk_data)

