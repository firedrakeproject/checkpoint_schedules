---
title: 'checkpoint_schedules: schedules for incremental of adjoint simulations.'

tags:
  - Python
  - Adjoint-based gradient
  - Checkpointing method
  - Revolvers
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

Adjoint-based gradient problems are commonly used to compute of sensitivity measures for topology optimization `[@papadopoulos2021computing]`, inverse problems [@Plessix2006], flow optimisation, and control [@jansen2011adjoint]. These physical problems are governed by time-dependent partial differential equations, which are referred here to as forward problems. Hence, the adjoint-based sensitivity leads to computations of the time-dependent adjoint-based gradients are characterised by their backward progression in time and their dependence on the solution of a forward solution. In such cases, one must require either store the forward data or recompute the forward problem to obtain essential data for the adjoint computations.

Storing the forward data at every time step in RAM (Random Access memory) is impractical for large systems due to the potential for high memory usage during extensive time executions. Techniques such as storing the forward data on disk or employing a checkpointing method can overcome this memory usage issue. Checkpointing strategies involve creating a sequence of schedules that coordinate the selective storage of forward data at specific time steps. During the adjoint computation, these schedules provide instructions for restarting the forward solver from the nearest time step where forward data was stored, in case the forward data is unavailable, until the step where the adjoint needs to be computed. 


In order to optimise the checkpointing scheme, the revolvers algorithm aims to minimise the required forward recomputation for any allowed checkpoint storage and for any previously defined number of time steps [@griewank2000algorithm]. Also, when the number of time steps is a priori unknown [@wang2009minimal]. Optimal checkpointing strategies also take into consideration different storage types [@stumm2009multistage; @aupy2016optimal; @schanen2016asynchronous; @herrmann2020h].

# Statement of Need
That accomodates cases where the user wants to store the forward checkpoint data of all time-steps and not apply any checkpointing strategy given by revolvers methods. These schedules are available to simplify the integration with a adjoint-based gradient solver not restricted to using a checkpointing based on revolver methods. 
In general, the Revolve algorithms provide a schedule for the case where time-step for storing the required data to restart the forward solver. However, this approach does not fully address the general adjoint-based gradient problem since the data required to restart a forward calculation differs from the data used for the adjoint computation. *checkpoint_schedules* is a Python package that has been developed to tackle this issue, offering a schedule with a sequence of actions that provide functionalities of forward or adjoint advancement over the entire interval of steps, forward and adjoint data storage and retrieval. The schedule explicitly incorporates the data buffering in intermediate storage of the forward daya, ensuring that forward variables can be defined and computed by the forward solver before the storage. Furthermore, the schedule distinguishes between the storage of forward restart data and forward data required in the adjoint computation. 

This package is designed to be flexible in interpreting and converting designs from various existing strategies. It is already functioning with the following checkpoint approaches:
* Revolve [@griewank2000algorithm]; 
* MultiStage checkpointing [@stumm2009multistage];
* Disk revolve [@aupy2016optimal];
* Periodic Disk Revolve [@aupy2017periodicity];
* H-Revolve [@herrmann2020h];
* two-level mixed periodic/binomial checkpointing [@goldberg2020bathymetric; @pringle2016providing]
* Mixed storage of forward restart and non-linear dependency data [@maddison2023implementation];

In addition to the checkpointing schedules mentioned above, checkpoint_schedules also offers two additional types of checkpointing schedules for the cases where no checkpointing strategy is used. In this case, the forward data is stored every time step, either in RAM or on disk. Also, there is a checkpointing schedule available for situations where no adjoint calculation is required.

*checkpoint_schedules* provide generators, which enable easy iteration over the sequence of schedules for any above-mentioned checkpointing strategies. This allows users to efficiently access and use the desired checkpointing schedule based on their specific needs and computational requirements.

# Acknowledgments

# References
