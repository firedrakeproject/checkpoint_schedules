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

Adjoint-based gradient is a method employed for a variety of applications as topology optimisation [@papadopoulos2021computing], inverse problems [@Plessix2006], flow control [@jansen2011]. In such applications, it is usual deal with time-dependent partial differential equations (PDEs) to model the the physical problem, e.g., fluid flow, wave propagation. In this context, to compute the adjoint-based gradient requires solving an adjoint equation characterised by their backward time advancing and their dependence on the data obtained from the physcial govern equations, which in the current text is referred to as forward solvers.

Storing forward data at each time step in memory can be impractical for large systems, given the potential for high memory usage during the computational execution. Strategies of employing checkpointing aims to address this memory challenge. Optimal checkpointing strategies aim minimising the forward solver recomputation for a given number of steps to store the forward data (number of checkpoint storage). [@griewank2000algorithm] proposed an optimal strategy for any checkpoint storage and pre-defined time steps. [@wang2009minimal], on the other hand, proposed an optimal checkpointing for the cases where the number of time steps was a prior unkown. Checkpointing strategies have also presented for mixing the storage, i. e., checkpoint storage in memory or disk [@stumm2009multistage; @aupy2016optimal; @schanen2016asynchronous; @aupy2017periodicity; @herrmann2020; @maddison2023]. Furthermore, in the context of multi-stage time step discretisation [@Zhang_2023].

<!-- Statement of need -->
Integrating a single checkpoint strategy in an adjoint-based gradient solver that performs for distinct applications may not yield sufficient arrangement for every case. However, either writing or adapting different checkpointing approaches can lead to substantial effort since their implementations are usually originating from different codes ources. *checkpointing_schedules* package aims to overcome this issue with the flexibility to include different checkpointing strategies. This package can generate schedules based on the checkpointings for a variety of requirements, such as for mixed storage, for the cases where the forward time steps are a priori unknown, and for unlimited adjoint executions (online checkpointing). 

Essentialy, *checkpointing_schedules* generate schedules with functionalities that indicate the time advancing the forward and adjoint solvers, data storage and retrieval, and the solver finalisations. This package is able to provide schedules for the Revolve checkpointing [@griewank2000algorithm], and the checkpointing strategies with mixed storage: multistage checkpointing [@stumm2009multistage]; disk-revolve checkpointing [@aupy2016optimal]; periodic-disk revolve checkpointing [@aupy2017periodicity]; two-level checkpointing [@pringle2016providing], H-Revolve checkpointing [@herrmann2020]; and mixed storage checkpointing [@maddison2023]. 

*checkpointing_schedules* extends to its capabilities for the trivial checkpointing, which entails the forward data storage of all time steps without any checkpointing strategy. Furthermore, the package provides an alternative for the cases where no adjoint executions are needed. 

In conclusion, integrating the *checkpoint_schedules* package with an adjoint-based gradient solver is effective due to its condition of a variety of checkpointing strategies. Moreover, it is flexible in hosting new checkpointing methods, and it was built to be easy for using and integrating with other codes.

# Schedules

In essence, the schedules have functionalities for time advancing the forward and adjoint solvers, data storage and retrieval, and indicate the solvers finalisation. 


# Acknowledgments
This work was supported by the Engineering and Physical Sciences Research Council [EP/W029731/1 and EP/W026066/1]. J. R. M. is funded by ... 

# References
