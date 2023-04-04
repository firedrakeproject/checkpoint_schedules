from hrevolve_checkpointing import checkpoint_schedules, Forward, Backward, function


TS = 0
TF = 1
H = 0.05
S = int(TF/H)
SCHK = 3
I = 0
fwd = Forward(S)
fwd.def_equation()
bwd = Backward()
bwd.def_equation()
manage = checkpoint_schedules.Manage(fwd, bwd, function.Backend(), SCHK)
manage.actions()



