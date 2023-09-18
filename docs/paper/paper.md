---
title: 'checkpoint_schedules: schedules for incremental of adjoint simulations.'

tags:
  - Python
  - Checkpointing methods
  - Adjoint-based gradient
authors:
  - name: Daiane I. Dolci
    affiliation: 1
  - name: James R. Maddison
    affiliation: 2
  - name: David A. Ham
    affiliation: 1
affiliations:
 - name: Department of Mathematics, Imperial College London, London, SW72AZ, UK.
   index: 1
 - name: School of Mathematics and Maxwell Institute for Mathematical Sciences, The University of Edinburgh, EH9 3FD
   index: 2
date:  September 2023
bibliography: paper.bib
---

# Summary
*checkpointing_schedules* provides schedules for step based incremental
checkpointing of the adjoints to computer models. The schedules contain
instructions indicating the sequence of forward and adjoint steps to be
executed, and the data storage and retrieval to be performed. These
instructions are independent of the model implementation, which enables the
model authors to switch between checkpointing algorithms without recoding.
Conversely, *checkpointing_schedules* provides developers of checkpointing
algorithms a direct mechanism to convey their work to model authors.
*checkpointing_schedules* has been integrated into **tlm_adjoint** [@tlm2019],
a Python library designed for the automated derivation of higher-order
tangent-linear and adjoint models and work is ongoing to integrate it with
**pyadjoint** [@Mitusch2019]. This package can be incorporated into other gradient
solvers based on adjoint methods, regardless of the specific approach taken to
generate the adjoint model.

The use of adjoint calculations to compute the gradient of a quantity of
interest resulting from the solution of a system of partial differential
equations (PDEs) is widespread and well-established. The resulting gradient may
be employed for many purposes, include topology
optimisation [@papadopoulos2021computing], inverse problems [@Plessix2006],
flow control [@Jansen2011]. 

Solving the adjoint to a non-linear time-dependent PDE requires the forward PDE
to be solved first. The adjoint PDE is then solved in a reverse time
order, but depends on the forward state. Storing the entire forward state in
preparation for the adjoint calculation has a memory footprint linear in the
number of time steps. For sufficiently large problems this will exhaust the
memory of any computer system. 

In contrast, checkpointing approaches store only the state required to restart
the forward calculation from a limited set of steps. As the adjoint
calculation progresses, the forward computation is progressively rerun from the
latest available stored state up to the current adjoint step. This enables less
forward state to be stored, at the expense of a higher computational cost as
forward steps are run more than once. [@griewank2000algorithm] proposed a
checkpointing algorithm which is optimal under certain assumptions, including
that the number of steps is known in advance, and that all the storage has
equal access cost. Subsequent authors have produced checkpointing algorithms
that relax these requirements in various ways, such as by accounting for
different types of storage (e.g. RAM and disk) or by not requiring the number
of steps to be known in advance, for example [@stumm2009multistage;
@aupy2016optimal; @schanen2016; @aupy2017periodicity;
@herrmann2020; @maddison2023; @Zhang_2023]. 

# Statement of need

This situation is typical across computational mathematics: there exists a
diversity of algorithms whose applicability and optimality depends on the
nature and parameters of the problem to be solved. From the perspective of
users who wish to construct adjoint solvers this creates the need to swap out
different checkpointing algorithms in response to changes in the equations,
discretisations, and computer systems with which they work. Those users will
often lack the expertise or the time to continually reimplement additional
algorithms in their framework. Further, such reimplementation work is wasteful
and error-prone. 

*checkpointing_schedules* provides a number of checkpointing algorithms
accessible through a common interface and and these are interchangeable without recoding.
This can be used in conjunction with an adjoint framework such as tlm_adjoint
or pyadjoint and a compatible PDE framework, such as Firedrake [@FiredrakeUserManual] 
or FEniCS [@AlnaesEtal2015] to enable users to create adjoint solvers for their choice
of PDE, numerical methods, and checkpointing algorithm all without recoding the
underlying algorithms.

Some of the algorithms supported by *checkpointing_schedules* have been
implemented many times, while for others, such as H-Revolve the only previously
published implementation was a simple proof of concept in the original paper
[@herrmann2020]. The checkpoint schedule API provided by *checkpoint_schedules*
further provides a toolkit for the implementation of further checkpoint
schedules, thereby providing a direct route from algorithm developers to users.

# Software description
Currently, *checkpoint_schedules* is able to generate schedules for the
following checkpointing schemes: Revolve [@stumm2009multistage]; disk-revolve
[@aupy2016optimal]; periodic-disk revolve [@aupy2017periodicity];  two-level
[@pringle2016providing]; H-Revolve [@herrmann2020]; and mixed storage
checkpointing [@maddison2023]. It also contains trivial schedules which store
the entire forward state. This enables users to support adjoint calculations
with or without checkpointing via a single code path. 

The complete documentation for *checkpoint_schedules* is available at https://www.firedrakeproject.org/checkpoint_schedules/.

# Acknowledgments
This work was supported by the Engineering and Physical Sciences Research
Council [EP/W029731/1 and EP/W026066/1]. J. R. M. was supported by the Natural
Environment Research Council [NE/T001607/1].

# References
