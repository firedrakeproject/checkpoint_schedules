Introduction
============
Checkpointing is a technique used to manage memory usage during adjoint-based gradient calculations and can 
be essential in various scientific applications, including sensitivity analyses in fluids, inverse problems, and topology optimization.

The checkpoint_schedules package determines when to store and load checkpoint data, as well as when to advance the forward or adjoint computations. 