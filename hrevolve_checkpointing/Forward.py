import sympy as sp

class Forward():
    """This object define the a forward solver.

    """
    def __init__(self, steps):
        self.exp = None
        self.chk = None
        self.output_1 = None 
        self.n_0 = None
        self.steps = steps


    def def_equation(self):
        """Define the symbolic equation.

        """
        # Create a symbol x
        x = sp.symbols("x")
        self.exp = x + 1
      


    def advance(self, n_0: int, n_1: int) -> None:
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        for s in range(n_0, n_1):
            self.n_0 = n_0
            out = self.exp.subs("x", s)
            print(n_1, self.steps)
            if s==self.n_0 or n_1==self.steps:
                self.get_checkpoint(out)



    def get_timesteps(self) -> int:
        """Return time steps.

        """
        return self.steps
   


    def get_checkpoint(self, forward_output) -> None:
        """Get the output of the forward solver.

        """
        self.chk = forward_output
    
    def get_initial_condition():
        print("here")

            
    # def checkpointing(self, n_write) -> None:
    #     """Verify if is checkpointed.
        
    #     Parameters
    #     ----------
    #     is_checkpointed
    #         If `True`, the checkpoint data is saved.

    #     """ 
    #     assert n_write==self.n_0
    #     self.func.save_checkpoint(self.chk)
    
    # def save_checkpoint(self, data) -> None:
    #     """Append the checkpoint data.

    #     """
    #     self.chk_data.append(data)
    
    # def get_checkpoint_id(self, n_write: int) -> None:
    #     """Collect the checkpoint identity.
    
    #     """
    #     self.chk_id = n_write


