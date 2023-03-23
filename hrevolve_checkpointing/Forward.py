import sympy as sp
from hrevolve_checkpointing.Function import Function

class forward():
    def __init__(self, y_init, h, final_t):
        self.y_init = y_init  # Initial condition 
        self.h = h
        self.final_t = final_t
        self.equation = None


    def def_equation(self):
        # Create a symbol x
        x = sp.symbols("x")
        self.equation = sp.Eq(x-1, 0)

    def advance(self, n0, n1):
        for s in range(n0, n1):
            solution = sp.solveset(self.equation, s)
            print(solution)


    def checkpointing(self, is_checkpointed):
        if is_checkpointed:
            Function.save_checkpoint()
