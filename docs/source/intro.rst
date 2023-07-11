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

Concerned with the optimal checkpointing strategy, the revolve algorithm ask to minimize the number of forward steps taken and then, given that the number of forward steps is minimal, 
further minimizes the number of times a checkpoint is stored. The revolve algorithm provides an optimal schedule for the case where the number of forward steps is known ahead of the 
calculation, and where checkpoints store data required to restart the forward but not necessarily data required to advance the adjoint, which does not attend enterly the general 
adjoint-based gradient problem since the data required to restart a forward calculation, and the data required by the adjoint, differ. Concerning with this issue, the *checkpoint_schedules* is a Python package that provide schedule that the forward calculation divided into a known sequence of steps, and considers the 
problem of defining a schedule with advancement of the forward or adjoint over full steps. The schedule explicitly incorporates the buffering of data in an “intermediate storage”, 
ensuring that forward variables can be defined and computed by the forward before storage in a checkpoint. The schedule distinguishes between storage of forward restart and non-linear 
dependency data. Additionaly, *checkpoint_schedules* is flexible to be applied to a number of existing approaches, including the revolve algorithm. It is already working for the multistage 
approach of Stumm and Walther [10], the two-level mixed periodic/binomial approach described in Pringle et al. [21] and H-Revolve schedules [13].

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


