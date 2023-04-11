from hrevolve_checkpointing import *

t_s = 0
t_f = 1
h = 0.05
initial_condition = 0
s = int(t_f/h)
schk = 3
fwd = Forward(s, initial_condition)
fwd.DefEquation()
bwd = Backward()
bwd.DefEquation()
f = Function.Backend(schk)
manage = checkpoint_schedules.Manage(fwd, bwd, f, schk)
manage.actions()



