from checkpoint_schedules import Forward, Backward, Manage

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
manage = Manage(fwd, bwd, schk)
manage.actions()