from . import \
    (HRevolveCheckpointSchedule, Write, Clear, Configure, 
     Forward, EndForward, Reverse, Read, EndReverse)
import functools

__all__ = \
    [
        "Manage"
    ]


class Manage():
    """This object manage the checkpointing.

    """
    def __init__(self, forward, reverse, checkpoint_function, save_chk):
        self.save_chk = save_chk
        self.forward = forward
        self.reverse = reverse
        self.chk_function = checkpoint_function

    def actions(self):
        n = self.forward.GetTimesteps()
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")
        
        @action.register(Clear)
        def action_clear(cp_action):
            if cp_action.clear_ics:
                ics.clear()
            if cp_action.clear_data:
                data.clear()

        @action.register(Configure)
        def action_configure(cp_action):
            nonlocal store_ics, store_data

            store_ics = cp_action.store_ics
            store_data = cp_action.store_data
            
        @action.register(Write)
        def action_write(cp_action):
            assert len(ics) > 0 or len(data) > 0
            if len(ics) > 0:
                if len(data) > 0:
                    assert cp_action.n == min(min(ics), min(data))
                else:
                    assert cp_action.n == min(ics)
            elif len(data) > 0:
                assert cp_action.n == min(data)

            snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n

            # Start at the current location of the forward
            assert model_n is not None and model_n == cp_action.n0

            if hrev_schedule.max_n() is not None:
                # Do not advance further than the current location of the adjoint
                assert cp_action.n1 <= n - model_r
            n1 = min(cp_action.n1, n)

            if store_ics:
                # No forward restart data for these steps is stored
                assert len(ics.intersection(range(cp_action.n0, n1))) == 0

            if store_data:
                # No non-linear dependency data for these steps is stored
                assert len(data.intersection(range(cp_action.n0, n1))) == 0

            model_n = n1
            if store_ics:
                ics.update(range(cp_action.n0, n1))
            if store_data:
                data.update(range(cp_action.n0, n1))
            if n1 == n:
                hrev_schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            nonlocal model_r

            # Start at the current location of the adjoint
            assert cp_action.n1 == n - model_r
            # Advance at least one step
            assert cp_action.n0 < cp_action.n1
            # Non-linear dependency data for these steps is stored
            assert data.issuperset(range(cp_action.n0, cp_action.n1))

            model_r += cp_action.n1 - cp_action.n0

        @action.register(Read)
        def action_read(cp_action):
            nonlocal model_n

            # The checkpoint exists
            assert cp_action.n in snapshots[cp_action.storage]

            cp = snapshots[cp_action.storage][cp_action.n]

            # No data is currently stored for this step
            assert cp_action.n not in ics
            assert cp_action.n not in data
            # The checkpoint contains forward restart or non-linear dependency data
            assert len(cp[0]) > 0 or len(cp[1]) > 0

            # The checkpoint data is before the current location of the adjoint
            assert cp_action.n < n - model_r

            model_n = None

            if len(cp[0]) > 0:
                ics.clear()
                ics.update(cp[0])
                model_n = cp_action.n

            if len(cp[1]) > 0:
                data.clear()
                data.update(cp[1])

            if cp_action.delete:
                del snapshots[cp_action.storage][cp_action.n]

        @action.register(EndForward)
        def action_end_forward(cp_action):
        
            # The correct number of forward steps has been taken
            assert model_n is not None and model_n == n
           
        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            nonlocal model_r

            # The correct number of adjoint steps has been taken
            assert model_r == n

            if not cp_action.exhausted:
                model_r = 0

        S = (self.save_chk,)
        for s in S:
            model_n = 0
            model_r = 0

            store_ics = False
            ics = set()
            store_data = False
            data = set()

            fwd_chk = None
            snapshots = {"RAM": {}, "disk": {}}

            # H revolve schedule
            steps = self.forward.GetTimesteps()
            hrev_schedule = HRevolveCheckpointSchedule(steps, self.save_chk, 0)
        
            if hrev_schedule is None:
                print("Incompatible with schedule type")
    
            assert hrev_schedule.n() == 0
            assert hrev_schedule.r() == 0
            assert hrev_schedule.max_n() is None or hrev_schedule.max_n() == n
        
            while True:
                cp_action = next(hrev_schedule)
                action(cp_action)
                if isinstance(cp_action, Write):
                    self.chk_function.StoreCheckpoint(self.forward.ic)
                elif isinstance(cp_action, Forward):
                    self.forward.Advance(cp_action.n0, cp_action.n1)
                elif isinstance(cp_action, Reverse):
                    self.reverse.Advance(cp_action.n1, cp_action.n0, self.forward.chk)
                elif isinstance(cp_action, Read):
                    fwd_chk = self.chk_function.GetCheckpoint()
                    self.forward.UpdateInitCondition(fwd_chk)
                    if cp_action.delete:
                        self.chk_function.DeleteCheckpoint()

                assert model_n is None or model_n == hrev_schedule.n()
                assert model_r == hrev_schedule.r()

                if isinstance(cp_action, EndReverse):
                    break
   
