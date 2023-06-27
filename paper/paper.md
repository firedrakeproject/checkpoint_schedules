---
title: 'checkpoint_schedules: '
tags:
  - Python
  - Adjoint-based gradients
  - Checkpointing method
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
date: 20 August 2020
bibliography: paper.bib


# Summary
* action_forward: This actions is used to advance the forward solver for a step number (s), and to configure the intermediate storage. The forward action given by checkpoint_schedulues package reads Forward(n0, n1, write_ics, write_adj_deps, storage), where n0 and n1 are respectivelly initial and final step, write_ics is used to indicate whether the data used to restart the forward solver, write_adj_deps indicates whether to save the data used in the adjoint computation, and storage indicates the level (either RAM or disk) to store the data. 

* action_reverse: This actions is used to execute the adjoint solver from the start step n0 to the step n1, and to clear the forward data that was used in the adjoint solver. The Reverse action given by checkpoint_schedules package is Reverse(n0, n1, clear_adj_deps), where n0 and n1 are respectivelly initial and final step, clear_adj_deps indicates whether to clear the forward data used in the adjoint computation.

The actions printed above are in general form described as follows:

* *Forward(n0, n1, write_ics, write_adj_deps, storage)*:

    - Executes the forward solver from step *n0* to step *n1*.
    - Write the forward data of step *n0* if *write_ics* is *True*.
    - Indicates whether to store the forward data for the adjoint computation (*write_adj_deps*).
    - Indicate the storage level for the forward data (storage).

* *Reverse(n0, n1, clear_adj_deps)*:

    - Executes the adjoint solver from step *n0* to step *n1*.
    - Clears the adjoint dependencies (*adj_deps*) used in the adjoint computation.

* *Copy(n, from_storage, to_storage, delete)*:

    - Copy the forward data related to step n from one storage location (*from_storage*) to another storage location (*to_storage*).
    - Indicate whether to delete the copied data from the source storage location (delete).

* *EndForward()*:
    - Indicates the finalization of the forward solver.

* *EndReverse()*:

    - Indicate the finalisation of the adjoint solver.
