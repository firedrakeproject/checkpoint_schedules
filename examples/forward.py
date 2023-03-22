from hrevolve_checkpointing.checkpoint_schedules import manager

import matplotlib.pyplot as plt
import numpy as np
import functools
# Initialization 
class forward():
    def __init__(self, y_init, h, final_t):
        self.y_init = y_init  # Initial condition 
        self.h = h
        self.final_t = final_t

    # Function that returns dx/dt 
    def mydiff(self):
        c = 4 
        alpha = 3 
        ym = []
        y = self.y_init
        t = 0
        while t < self.final_t :
            F = c * np.cos(alpha * t)
            sol = y + self.h * F
            t += self.h
            ym.append(sol)
            y = sol
        return ym

    # def write_action():

    def plotting(self, t, y):
        plt.plot(t, y)
        plt.title('solution') 
        plt.xlabel('t')
        plt.ylabel('y(t)') 
        # plt.legend(["x1", "x2"]) 
        plt.grid()
        plt.show()


tstart = 0 
tstop = 1 
increment = 0.05
steps = int(tstop/increment)
save_chk = 2
# t = np.arange(tstart, tstop, increment)
# y_init = 0
# fwd = forward(y_init, increment, tstop)
# Solve ODE
# y = fwd.mydiff()
manager = manager(steps, save_chk)
actions = manager.actions()
