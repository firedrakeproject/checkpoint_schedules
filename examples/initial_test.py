from hrevolve_checkpointing.checkpoint_schedules import manage
from hrevolve_checkpointing.Forward import forward

import matplotlib.pyplot as plt
import numpy as np
import functools




tstart = 0 
tstop = 1 
increment = 0.05
steps = int(tstop/increment)
save_chk = 3
# t = np.arange(tstart, tstop, increment)
y_init = 0
fwd = forward(y_init, increment, tstop)
# Solve ODE
# y = fwd.mydiff()
manage = manage(steps, save_chk)
manage.forward = forward
forward.def_equation
forward.advance(0, 0, 10)

actions = manage.actions()





# def mydiff(self):
#     c = 4 
#     alpha = 3 
#     ym = []
#     y = self.y_init
#     t = 0
#     while t < self.final_t :
#         F = c * np.cos(alpha * t)
#         sol = y + self.h * F
#         t += self.h
#         ym.append(sol)
#         y = sol
#     return ym


# def plotting(self, t, y):
#     plt.plot(t, y)
#     plt.title('solution') 
#     plt.xlabel('t')
#     plt.ylabel('y(t)') 
#     # plt.legend(["x1", "x2"]) 
#     plt.grid()
#     plt.show()