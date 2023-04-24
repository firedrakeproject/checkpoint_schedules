#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Add corect license text
from .schedule import CheckpointSchedule, Forward, Reverse, \
    Read, Write, EndForward, EndReverse

__all__ = \
    [
        "PeriodicDiskCheckpointSchedule"
    ]


class PeriodicDiskCheckpointSchedule(CheckpointSchedule):
    def __init__(self, period):
        if period < 1:
            raise ValueError("period must be positive")

        super().__init__()
        self._period = period

    def iter(self):
        # Forward

        while self._max_n is None:
            yield Configure(True, False)
            if self._max_n is not None:
                # Unexpected finalize
                raise RuntimeError("Invalid checkpointing state")
            n0 = self._n
            n1 = n0 + self._period
            self._n = n1
            yield Forward(n0, n1)

            # Finalize permitted here

            yield Write(n0, "disk")
            yield Clear(True, True)

        yield EndForward()

        while True:
            # Reverse

            while self._r < self._max_n:
                n = self._max_n - self._r - 1
                n0 = (n // self._period) * self._period
                del n
                n1 = min(n0 + self._period, self._max_n)
                if self._r != self._max_n - n1:
                    raise RuntimeError("Invalid checkpointing state")

                self._n = n0
                yield Read(n0, "disk", False)
                yield Clear(True, True)

                yield Configure(False, True)
                self._n = n1
                yield Forward(n0, n1)

                self._r = self._max_n - n0
                yield Reverse(n1, n0)
                yield Clear(True, True)
            if self._r != self._max_n:
                raise RuntimeError("Invalid checkpointing state")

            # Reset for new reverse

            self._r = 0
            yield EndReverse(False)

    def is_exhausted(self):
        return False

    def uses_disk_storage(self):
        return True
