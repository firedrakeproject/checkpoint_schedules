.. _introduction:

Introduction
============
The time-dependent adjoint-based gradient is characterised by its backward progression in time and its dependence on the solution of a forward system (cite). 
Consequently, the time-dependent adjoint requires either storing the forward data or recomputation of the forward problem to obtain essential data for the adjoint computations.

Storing the forward data at every time step is impractical for large systems, as it can result in high memory usage for extensive time executions.
This challenge can be overcome by employing the checkpointing method, which efficiently manages memory usage.
To determine an optimal checkpointing strategy, the revolvers algorithm aims to minimize the number of forward steps taken. Once the minimal number 
of forward steps is established, the algorithm further minimizes the frequency of checkpoint storage. 
The revolvers algorithm provides an optimal schedule in scenarios where the number of forward steps is known prior to the calculation, 
and where checkpoints store the necessary data for restarting the forward computation.


In summary, checkpointing strategies involve creating a sequence of schedules that coordinate the selective 
storage of forward data at specific time steps. During the adjoint computation, these schedules provide instructions for restarting 
the forward solver from the nearest time step where forward data was stored, in case the forward data is unavailable, until the step 
where the adjoint needs to be computed.

The *checkpoint_schedules* is a Python package that offers schedules given by a sequence of actions that provides 
functionalities of forward or adjoint advancement over the entire interval of steps, data storage, and data retrieval.
The *checkpoint_schedules* package explicitly incorporates data buffering in an intermediate storage, ensuring that forward 
variables can be defined and computed before storage in a checkpoint. Furthermore, the schedule distinguishes 
between the storage of forward restart data and the forward data required for the adjoint computation. The *checkpoint_schedules* package 
is flexible to interpret and convert designs from various existing approaches, including the revolve algorithm. It has already been 
successfully implemented for the multistage approach of Stumm and Walther (20XX), the two-level mixed periodic/binomial approach described 
in Pringle et al. (20XX), and H-Revolve schedules (Smith et al., 20XX). 

To advance in the using of checkpoint_schedules package in the adjoint-based gradient computation, we recommend to read the following section:
:ref:`example <example_checkpoint_schedules>`


