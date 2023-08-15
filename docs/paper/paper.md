---
title: 'checkpoint_schedules: schedules for incremental of adjoint simulations.'

tags:
  - Python
  - Checkpointing methods
  - Adjoint-based gradient
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
date:  August 2023
bibliography: paper.bib

---
# Summary

Adjoint-based gradient is a method employed for a range of applications as topology optimisation [@papadopoulos2021computing], inverse problems [@Plessix2006], flow control [@jansen2011]. These different problems are governed by time-dependent partial differential equations (PDEs), which are here referred to as forward solvers. In such cases, the adjoint-based sensitivity requires the computation of time-dependent adjoint equations characterised by their backward time advancing and their dependence on the data obtained from the forward solver. Consequently, the adjoint-based gradient computation requires to execute the forward solver, and then either store or recompute the forward data.

Storing forward data at each time step in memory RAM can be impractical for large systems, given the potential for high memory usage during the time executions. Strategies of employing checkpointing strategies have been employed to address this memory challenge. 

Checkpointing methods involve generating schedules that coordinate forward and adjoint time advancing, data storage and retrieval at specific time points. Optimal checkpointing strategies works around minimising the forward solver recomputing for a given predefined number of steps to store the forward data (number of checkpoint storage). [@griewank2000algorithm] proposed an optimal strategy for any checkpoint storage and a pre-defined time step. [@wang2009minimal], on the other hand, an optimal checkpointing strategy for the cases where the number of time steps was a pior unkown. Additionally, checkpointing strategies have been proposed for mixed storage, i. e.,  forward data stored in RAM and disk[@stumm2009multistage; @aupy2016optimal; @schanen2016asynchronous; @aupy2017periodicity; @herrmann2020; @maddison2023]. Also, in the context of multi-stage time step discretisation [@Zhang_2023].


# Statement of Need
Different checkpointing strategies have been proposed in the literature, consequently different implementations have been developed to each strategy. Therefore, to incorporate a checkpoint package in 
a adjoint-based gradient code requires a significant amount of time either to implement the checkpointing strategy or adapt an existing implementation to the code. Additionally, if the a code uses the adjoint-based gradient for different applications, or different time discretisations, and the checkpointing strategy may needs to be modified or adapted to the new code. 

The *checkpointing_schedules* aims to supply this issue with designed package that is flexible in interpreting and incorporating distingues checkpointing strategies. *checkpoint_schedules* builds the schedules that in essence provide functionalities to time advancing the forward and adjoint solvers, checkpoint data storage and retrieval, and indicate the solver finalisations. The schedules ready incorporated the Revolve checkpointing [@griewank2000algorithm], and the checkpoint strategies for mixed storage type [@stumm2009multistage, @aupy2016optimal; @aupy2017periodicity, @pringle2016providing, herrmann2020, @maddison2023]. 

This package extends to its capabilities the solver executions through trivial checkpointing, which entails the forward data storage of all time steps without any checkpointing method. Furthermore, the package provides an alternative for the cases where no adjoint executions are needed, excluding any forward data storage and retrieval. That makes easier to incorporate the package in a adjoint-based gradient code that is not restricted to a specific checkpointing strategy and exclusively execute an adjoint solver.

In conclusion, *checkpoint_schedules* package works as a flexible tool to incorporate checkpointing strategies ready available and also new strategies. The schedules accept online and offline checkpointing strategies, and it is easy to incorporate checkpointing strategies that the the forward time steps is a priori unknown. For additional details, please refer to the package documentation available at


# Acknowledgments
This work was supported by the Engineering and Physical Sciences Research Council [EP/W029731/1 and EP/W026066/1]. J. R. M. is funded by ... 

# References
