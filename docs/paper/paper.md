---
title: 'checkpoint_schedules: schedules for incremental of adjoint simulations.'

tags:
  - Python
  - Adjoint-based gradient
  - Checkpointing method
  -Revolvers
# authors:
#   - name: 
#     affiliation: 1
#   - name: 
#     affiliation: 1
#   - name: 
#     affiliation: 2
affiliations:
#  - name: .
#    index: 1
#  - name:  
#    index: 2
date:  July 2023
bibliography: paper.bib

---
# Indroduction

Adjoint-based gradient problems are commonly used to compute of sensitivity measures for topology optimization @papadopoulos2021computing, inverse problems @Plessix:2006, flow optimisation, and control @jansen2011adjoint. These phisical problems are governed by time-dependent partial differential equations, which are referred here to as forward problem. Hence, the adjoint-based sensitivity leads to computations of the time-dependent adjoint-based gradients that are characterised by their backward progression in time and their dependence on the solution of a forward solutions. In the such cases, one must requires either store the forward data or recompute the forward problem to obtain essential data for the adjoint computations.

Storing the forward data at every time step in RAM (Random Access memory) is impractical for large systems, due to the potential for high memory usage during extensive time executions. Techniques as storing the forward data on disk or employ a checkpointing method can overcome this memory usage issue. Checkpointing strategies involve creating a sequence of schedules that coordinate the selective storage of forward data at specific time steps. During the adjoint computation, these schedules provide instructions for restarting the forward solver from the nearest time step where forward data was stored, in case the forward data is unavailable, until the step where the adjoint needs to be computed. 

To optimise the checkpointing, the revolve algorithmic aims to minimise the number of forward steps storage and then, given that the number of forward steps is minimal, further minimises the number of times a checkpoint is stored. 

# Statement of Need

Concerning with this issue, the *checkpoint_schedules* is a Python package that provide schedule that the forward calculation divided into a known sequence of steps, and considers the 
problem of defining a schedule with advancement of the forward or adjoint over full steps. The schedule explicitly incorporates the buffering of data in an “intermediate storage”, 
ensuring that forward variables can be defined and computed by the forward before storage in a checkpoint. The schedule distinguishes between storage of forward restart and non-linear 
dependency data. Additionaly, *checkpoint_schedules* is flexible to be applied to a number of existing approaches, including the revolve algorithm. It is already working for the multistage  approach of Stumm and Walther [10], the two-level mixed periodic/binomial approach described in Pringle et al. [21] and H-Revolve schedules [13].

The revolve algorithms provides an optimal schedule for the case where the number of forward steps is known ahead of the calculation, and where checkpoints store data required to restart the forward but not necessarily data required to advance the adjoint, which does not attend enterly the general adjoint-based gradient problem since the data required to restart a forward calculation, and the data required by the adjoint, differ. 

In the adjoint-based gradient computation, the forward data used as initial conditions for the forward solver 
restarting can differ from the forward data required for the adjoint computation, e. g., for non-linear 
problems. Thus, we propose the *checkpoint_schedules* Python package the schedule distinguishes between the storage of forward restart data and the forward data required for the adjoint 
computation. The *checkpoint_schedules* package offers schedules given by a sequence of actions that provides functionalities 
of forward or adjoint advancement over the entire interval of steps. The *checkpoint_schedules* package explicitly incorporates data
buffering in an intermediate storage, ensuring that forward variables can be defined and computed before storage in a checkpoint. 
Furthermore, the *checkpoint_schedules* package is flexible to interpret and convert designs from various existing approaches, 
including the revolve algorithm. It has already been successfully working with the multistage approach [2], the two-level mixed 
periodic/binomial [6], and H-Revolve schedules [4]. 

The *checkpoint_schedules* also provides a generator able to iterate over the schedules and return the next action to perform. To advance 
in the basics of *checkpoint_schedules* employment and the generator usage, we recommend to read the following 
:ref:`section <example_checkpoint_schedules>`.



The *checkpoint_schedules* is a Python package that offers schedules given by a sequence of actions that provides 
functionalities of forward or adjoint advancement over the entire interval of steps, data storage, and data retrieval.
The *checkpoint_schedules* package explicitly incorporates data buffering in an intermediate storage, ensuring that forward 
variables can be defined and computed before storage in a checkpoint. Furthermore, the schedule distinguishes 
between the storage of forward restart data and the forward data required for the adjoint computation. The *checkpoint_schedules* package 
is flexible to interpret and convert designs from various existing approaches, including the revolve algorithm. It has already been 
successfully implemented for the multistage approach of Stumm and Walther (20XX), the two-level mixed periodic/binomial approach described 
in Pringle et al. (20XX), and H-Revolve schedules (Smith et al., 20XX). 

# Design

# Acknowledgments