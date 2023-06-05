from enum import Enum

class RevolverMethod(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    HREVOLVE = 0
    DISKREVOLVE = 1
    PERIODICDISKREVOLVE = 2

    def parameter(self, max_n, one_level_slot=None, two_level_slots=None, f_cost=1, b_cost=1, r_cost=0, w_cost=0):
        """_summary_

        Args:
            method (_type_): _description_
        """
        if self.name == 'DISKREVOLVE':
            function = disk_revolve
            n_slots = kwargs.get("n_slots")
            cost_fwd = kwargs.get("cost_fwd")
            cost_read = kwargs.get("cost_read")
            cost_write = kwargs.get("cost_write")

a = RevolverMethod(1).parameter()
print(a)
