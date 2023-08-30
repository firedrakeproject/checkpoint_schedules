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
The *checkpointing_schedules* provides schedules for step based incremental checkpointing of the adjoints to computer models. The schedules contain instructions indicating the sequence of forward and adjoint steps to be executed, and the data storage and retrieval to be performed. These instructions are independent of the model implementation, which enables the model authors to switch between checkpointing algorithms without recoding. Conversely, *checkpointing_schedules* provides developers of checkpointing algorithms a direct mechanism to convey their work to model authors. *checkpointing_schedules* has been integrated into **tlm_adjoint**, a Python library designed for the automated derivation of higher-order tangent-linear and adjoint models and work is ongoing to integrate it with **pyadjoint**. This package can be incorporated into other gradient solvers based on adjoint methods, regardless of the specific approach taken to generate the adjoint model.


# Statement of need
Adjoint-based gradient is a method employed for a variety of applications as topology optimisation [@papadopoulos2021computing], inverse problems [@Plessix2006], flow control [@jansen2011]. In such applications, it is usual deal with time-dependent partial differential equations (PDEs) to model the the physical problem, e.g., fluid flow, wave propagation. In this context, to compute the adjoint-based gradient requires solving an adjoint equation characterised by their backward time advancing and their dependence on the data obtained from the physcial govern equations, which in the current text is referred to as forward solvers.

Storing forward data at each time step in memory can be impractical for large systems, given the potential for high memory usage during the computational execution. Strategies of employing checkpointing aims to address this memory challenge. Optimal checkpointing strategies aim minimising the forward solver recomputation for a given number of steps to store the forward data (number of checkpoint storage). [@griewank2000algorithm] proposed an optimal strategy for any checkpoint storage and pre-defined time steps. Checkpointing strategies have also presented for mixing the storage, i. e., checkpoint storage in memory or disk [@stumm2009multistage; @aupy2016optimal; @schanen2016asynchronous; @aupy2017periodicity; @herrmann2020; @maddison2023]. Furthermore, in the context of multi-stage time step discretisation [@Zhang_2023].


Integrating a checkpointing strategies in an adjoint-based gradient solver performing for distinct applications may not yield sufficient arrangement for every case. However, either writing or adapting different checkpointing approaches can lead to substantial effort since their implementations are usually originating from different codes sources. *checkpointing_schedules* package aims to overcome this issue with the flexibility to include different checkpointing strategies. This package can generate schedules based on the checkpointings for a variety of requirements, such as for mixed storage, for the cases where the forward time steps are a priori unknown, and for unlimited adjoint executions (online checkpointing). 

Integrating the *checkpoint_schedules* package with an adjoint-based gradient solver is effective due to its condition of a variety of checkpointing strategies. Moreover, it is flexible in hosting new checkpointing methods.

# Software description
Currently, *checkpoint_schedules* is able to generate schedules for the following checkpointing schemes: Revolve [@stumm2009multistage]; disk-revolve [@aupy2016optimal]; periodic-disk revolve [@aupy2017periodicity];  two-level [@pringle2016providing]; H-Revolve [@herrmann2020]; and mixed storage checkpointing [@maddison2023]. In addition, it extends to its capabilities a trivial schedule for forward computation and for all storage data. 

The *checkpoint_schedules* repository contains documentions and notebook available at



# Acknowledgments
This work was supported by the Engineering and Physical Sciences Research Council [EP/W029731/1 and EP/W026066/1]. J. R. M. was supported by the Natural Environment Research Council
[NE/T001607/1].

This research was funded in whole, or in part, by the Natural Environment
Research Council [NE/T001607/1]. For the purpose of open access, the author has
applied a creative commons attribution (CC BY) licence to any author accepted
manuscript version arising.

# References
