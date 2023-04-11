from hrevolve_checkpointing import Function, Forward, Backward, checkpoint_schedules

t_s = 0
t_f = 1
h = 0.05
fwd_ic = 0
bwd_ic = 20
s = int(t_f/h)
schk = 3
fwd = Forward(s, fwd_ic)
fwd.DefEquation()
bwd = Backward()
bwd.DefEquation()
f = Function()
manage = checkpoint_schedules.Manage(fwd, bwd, f, schk)
manage.actions()



