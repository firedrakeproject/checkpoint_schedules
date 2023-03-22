from ..checkpoint_schedules import \
    (HRevolveCheckpointSchedule, Write, Clear, Configure, 
     Forward, EndForward)
import functools

class manager():
    def __init__(self, steps, save_chk):
        self.steps = steps
        self.save_chk = save_chk

    def actions(self):
        print("aqui")
        n = self.steps
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
            # The checkpoint contains forward restart or non-linear dependency data

            assert len(ics) > 0 or len(data) > 0

            # The checkpoint location is associated with the earliest step for
            # which data has been stored
            if len(ics) > 0:
                if len(data) > 0:
                    assert cp_action.n == min(min(ics), min(data))
                else:
                    assert cp_action.n == min(ics)
            elif len(data) > 0:
                assert cp_action.n == min(data)

            snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))
            # print(snapshots)
        
        
        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n

            # Start at the current location of the forward
            assert model_n is not None and model_n == cp_action.n0

            if hrev.max_n() is not None:
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
                hrev.finalize(n1)

        
        # @action.register(EndForward)
        # def action_end_forward(cp_action):
        #     # The correct number of forward steps has been taken
        #     assert model_n is not None and model_n == n

        S = (self.save_chk,)
        for s in S:
            model_n = 0
            model_r = 0

            store_ics = False
            ics = set()
            store_data = False
            data = set()

            snapshots = {"RAM": {}, "disk": {}}
            hrev = HRevolveCheckpointSchedule(self.steps, self.save_chk, 0)
            print(hrev.__dict__)
            assert hrev.n() == 0
            assert hrev.r() == 0
            assert hrev.max_n() is None or hrev.max_n() == n
            
        
            while True:
                cp_action = next(hrev)
                action(cp_action)
                print(action)
                assert model_n is None or model_n == hrev.n()
                assert model_r == hrev.r()
                
                # action_clear(cp_action)
        # print(cp_action)
        # action(cp_action)

        # iter = hrev.iter()

        # print( HRevolveCheckpointSchedule.iter.action(10))
        # print(len(y), len(t))
        # fwd.plotting(t, y)
